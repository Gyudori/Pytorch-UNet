from pathlib import Path
import torch
from torchvision.utils import save_image
import numpy as np

from unet import UNet
from predict import mask_to_image
from utils.data_loading import BasicDataset
from utils.utils import get_prediction_debug_image


def get_model_names():
    predict_multiple_epochs = False
    if predict_multiple_epochs:
        start = 50
        end = 200
        step = 50
        epochs = range(start, end + 1, step)
        model_names = [f"checkpoint_epoch{epoch}.pth" for epoch in epochs]
    else:
        model_names = ["checkpoint_epoch50.pth"]

    return model_names


def get_output_dir(model_names, i, output_dir):
    if len(model_names) == 1:
        return output_dir

    output_dir2 = output_dir / model_names[i].split(".")[0]
    output_dir2.mkdir(exist_ok=True)
    return output_dir2


def main():
    net = UNet(n_channels=3, n_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device=device)

    root_dir = Path("dataset_floorplan/2025_03_13/test")
    input_dir = root_dir / "imgs"
    input_files = list(input_dir.glob("*.png"))
    input_files.sort()

    run_dir = Path(
        "runs2/2025-04-11_09-39-16_enable_augmentation_BS_1_LR_1e-05_SCALE_1.0"
    )
    model_dir = run_dir / "checkpoints"

    output_dir = root_dir / "masks"
    output_dir.mkdir(exist_ok=True)

    model_names = get_model_names()

    for i in range(len(model_names)):
        model_name = model_names[i]
        print(f"Using model {model_name}...")
        model_filepath = model_dir / model_name
        state_dict = torch.load(model_filepath, map_location=device)
        mask_values = state_dict.pop("mask_values", [0, 1])
        net.load_state_dict(state_dict)
        # net.eval()

        for input_filepath in input_files:
            print(f"Predicting {input_filepath.name}...")
            img = (
                BasicDataset.get_preprocessed_image(
                    filepath=input_filepath, mask_values=mask_values
                )
                .to(device=device)
                .unsqueeze(0)
            )

            with torch.autocast(
                device.type if device.type != "mps" else "cpu", enabled=True
            ):
                prediction = net(img)
                prediction = prediction.float().cpu()
                pred_mask = prediction.argmax(dim=1)

                result = mask_to_image(
                    pred_mask.squeeze(0).numpy(),
                    mask_values,
                )

                model_name_output_dir = get_output_dir(model_names, i, output_dir)
                output_filepath = model_name_output_dir / f"{input_filepath.stem}.gif"
                result.save(output_filepath)

                img = img.to(device="cpu")
                pred_mask = pred_mask.to(device="cpu")
                prediction_debug_image = get_prediction_debug_image(
                    img.squeeze(), prediction, pred_mask
                )

                debug_image_filepath = (
                    model_name_output_dir / f"{input_filepath.stem}_debug.png"
                )

                save_image(prediction_debug_image, debug_image_filepath)

                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
