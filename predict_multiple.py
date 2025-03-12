from pathlib import Path
from PIL import Image
import torch

from unet import UNet
from predict import predict_img, mask_to_image


def main():
    input_filepath = Path("data_floorplan/test1/floorplan_resized.png")

    img = Image.open(input_filepath)

    if img.mode == "RGBA":
        img = img.convert("RGB")

    net = UNet(n_channels=3, n_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device=device)

    epochs = [i for i in range(100, 1001, 100)]
    
    model_names = [f"checkpoint_epoch{epoch}.pth" for epoch in epochs]

    run_dir = Path("runs/Feb12_17-56-30_tiso_augmentation_LR_1e-08_BS_1_SCALE_1.0")
    model_dir = run_dir / "checkpoints"

    out_dir = run_dir / "test"
    out_dir.mkdir(exist_ok=True)
    
    img.save(out_dir / "input.png")

    for model_name in model_names:
        model_filepath = model_dir / model_name
        state_dict = torch.load(model_filepath, map_location=device)
        mask_values = state_dict.pop("mask_values", [0, 1])
        net.load_state_dict(state_dict)

        mask = predict_img(
            net=net,
            full_img=img,
            device=device,
        )

        result = mask_to_image(mask, mask_values)

        model_name = model_name.split(".")[0]
        output_filepath = out_dir / f"{model_name}_output.png"
        result.save(output_filepath)


if __name__ == "__main__":
    main()
