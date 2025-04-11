import argparse
import logging
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate
from unet import UNet
from utils.data_loading import FloorplanDataset
from utils.dice_score import dice_loss
from utils.utils import predict_and_get_debug_image


def get_train_val_loader(
    val_percent: float,
    batch_size: int,
    img_scale: float,
):
    dataset_dir = Path("./dataset_floorplan/2025_03_13")
    dir_img = dataset_dir / "imgs"
    dir_mask = dataset_dir / "masks"

    dataset = FloorplanDataset(
        dir_img,
        dir_mask,
        img_scale,
        enable_augmentation=True,
        enable_degradation=False,
    )

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    num_workers = os.cpu_count()
    # num_workers = 1

    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    test_img_dir = dataset_dir / "test" / "imgs"
    test_mask_dir = dataset_dir / "test" / "masks"
    test_set = FloorplanDataset(
        test_img_dir, test_mask_dir, img_scale, enable_augmentation=False
    )
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

    return train_loader, val_loader, test_loader


def calculate_loss(
    model: nn.Module,
    images,
    true_masks,
    amp: bool,
    device,
    criterion,
):
    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        prediction = model(images)
        if model.n_classes == 1:
            loss = criterion(prediction.squeeze(1), true_masks.float())
            loss += dice_loss(
                F.sigmoid(prediction.squeeze(1)),
                true_masks.float(),
                multiclass=False,
            )
        else:
            loss = criterion(prediction, true_masks)
            loss += dice_loss(
                F.softmax(prediction, dim=1).float(),
                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                multiclass=True,
            )

    return loss


def update_weights(
    optimizer: optim.Optimizer,
    grad_scaler: torch.cuda.amp.GradScaler,
    model: nn.Module,
    loss: torch.Tensor,
    gradient_clipping: float,
):
    optimizer.zero_grad(set_to_none=True)
    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
    grad_scaler.step(optimizer)
    grad_scaler.update()


def validate(model, val_loader, device, amp, global_step, scheduler, writer, optimizer):
    val_score = evaluate(
        model,
        val_loader,
        device,
        amp,
    )
    scheduler.step(val_score)

    logging.info("Validation Dice score: {}".format(val_score))
    try:
        writer.add_scalar(
            "learning_rate",
            optimizer.param_groups[0]["lr"],
            global_step,
        )
        writer.add_scalar("validation/Dice", val_score, global_step)
    except Exception as e:
        print(e)
        pass


def test(model, test_loader, device, amp, epoch, validation_step, test_output_dir):
    for batch in test_loader:
        debug_image = predict_and_get_debug_image(
            model=model,
            batch=batch,
            device=device,
            amp=amp,
        )

        name = batch["name"][0]

        save_image(
            debug_image,
            test_output_dir
            / f"{name}_epoch_{epoch:03d}_val_step_{validation_step}.png",
        )


def step(
    model,
    images,
    true_masks,
    device,
    optimizer,
    amp,
    gradient_clipping,
    criterion,
    grad_scaler,
):
    assert images.shape[1] == model.n_channels, (
        f"Network has been defined with {model.n_channels} input channels, "
        f"but loaded images have {images.shape[1]} channels. Please check that "
        "the images are loaded correctly."
    )

    images = images.to(
        device=device,
        dtype=torch.float32,
        memory_format=torch.channels_last,
    )
    true_masks = true_masks.to(device=device, dtype=torch.long)

    loss = calculate_loss(
        model=model,
        images=images,
        true_masks=true_masks,
        amp=amp,
        device=device,
        criterion=criterion,
    )

    update_weights(
        optimizer=optimizer,
        grad_scaler=grad_scaler,
        model=model,
        loss=loss,
        gradient_clipping=gradient_clipping,
    )
    return loss


def train_model(
    model,
    device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_percent: float,
    img_scale: float,
    amp: bool = False,
    name: str = "",
    save_checkpoint: bool = True,
    weight_decay: float = 1e-2,
    gradient_clipping: float = 1.0,
):
    train_loader, val_loader, test_loader = get_train_val_loader(
        val_percent=val_percent,
        batch_size=batch_size,
        img_scale=img_scale,
    )
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)

    # (Initialize logging)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_name = f"{timestamp}"
    if name != "":
        log_name += f"_{name}"
    log_name += f"_BS_{batch_size}_LR_{learning_rate}_SCALE_{img_scale}"

    log_dir = Path("runs2") / log_name
    writer = SummaryWriter(log_dir=log_dir)
    logging.info(f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Testing size:    {n_test}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    """)

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=5
    )  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    validation_output_dir = Path(writer.log_dir) / "validation"
    validation_output_dir.mkdir(parents=True, exist_ok=True)
    validation_per_epoch = 5
    validation_interval = n_train // (validation_per_epoch * batch_size)

    test_output_dir = Path(writer.log_dir) / "test"
    test_output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                images, true_masks = batch["image"], batch["mask"]

                loss = step(
                    model=model,
                    images=images,
                    true_masks=true_masks,
                    device=device,
                    optimizer=optimizer,
                    amp=amp,
                    gradient_clipping=gradient_clipping,
                    criterion=criterion,
                    grad_scaler=grad_scaler,
                )

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                writer.add_scalar("Loss/train", loss.item(), global_step)
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                # Evaluation round
                if validation_interval > 0 and global_step % validation_interval == 0:
                    validate(
                        model=model,
                        val_loader=val_loader,
                        device=device,
                        amp=amp,
                        global_step=global_step,
                        scheduler=scheduler,
                        writer=writer,
                        optimizer=optimizer,
                    )

                    validation_step = global_step // validation_interval

                    test(
                        model,
                        test_loader,
                        device,
                        amp,
                        epoch,
                        validation_step,
                        test_output_dir,
                    )

        if save_checkpoint:
            dir_checkpoint = Path(writer.log_dir) / "checkpoints"
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict["mask_values"] = train_loader.dataset.dataset.mask_values
            torch.save(
                state_dict,
                str(dir_checkpoint / "checkpoint_epoch{}.pth".format(epoch)),
            )
            logging.info(f"Checkpoint {epoch} saved!")


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument(
        "--epochs", "-e", metavar="E", type=int, default=5, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        "-l",
        metavar="LR",
        type=float,
        default=1e-5,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--load", "-f", type=str, default=False, help="Load model from a .pth file"
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=1.0,
        help="Downscaling factor of the images",
    )
    parser.add_argument(
        "--validation",
        "-v",
        dest="val",
        type=float,
        default=10.0,
        help="Percent of the data that is used as validation (0-100)",
    )
    parser.add_argument(
        "--amp", action="store_true", default=False, help="Use mixed precision"
    )
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="Use bilinear upsampling"
    )
    parser.add_argument(
        "--classes", "-c", type=int, default=2, help="Number of classes"
    )
    parser.add_argument("--name", "-n", type=str, default="", help="Name of the run")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(
        f'Network:\n'
        f'\t{model.n_channels} input channels\n'
        f'\t{model.n_classes} output channels (classes)\n'
        f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling'
    )

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict["mask_values"]
        model.load_state_dict(state_dict)
        logging.info(f"Model loaded from {args.load}")

    model.to(device=device)
    try:
        train_model(
            model=model,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            name=args.name,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error(
            "Detected OutOfMemoryError! "
            "Enabling checkpointing to reduce memory usage, but this slows down training. "
            "Consider enabling AMP (--amp) for fast and memory efficient training"
        )
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
        )
