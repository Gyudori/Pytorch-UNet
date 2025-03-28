import torch
import torch.onnx
from pathlib import Path
from PIL import Image
import onnxruntime
import numpy as np

from unet import UNet
from predict import mask_to_image


def convert_to_onnx(onnx_path: Path):
    if onnx_path.exists():
        print(f"ONNX file already exists at {onnx_path}")
        return

    model = UNet(n_channels=3, n_classes=2)

    state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
    del state_dict["mask_values"]
    model.load_state_dict(state_dict)

    input_size = (3, 4096, 4096)

    model.eval()

    dummy_input = torch.randn(1, *input_size)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
    )

    print(f"Model converted to ONNX and saved at {onnx_path}")


def save_mask_channel(mask, channel):
    mask_channel = mask[channel]
    mask_channel_min = np.min(mask_channel)
    mask_channel_max = np.max(mask_channel)
    mask_channel = (mask_channel - mask_channel_min) / (
        mask_channel_max - mask_channel_min
    )
    mask_channel = (mask_channel * 255).astype(np.uint8)
    Image.fromarray(mask_channel).save(f"mask_{channel}.png")


if __name__ == "__main__":
    run_dir = Path("runs/Feb12_17-56-30_tiso_augmentation_LR_1e-08_BS_1_SCALE_1.0")
    run_dir = Path("runs/Mar25_13-46-20_tiso_LR_1e-08_BS_1_SCALE_1.0")
    epoch = 200
    weight_path = run_dir / f"checkpoints/checkpoint_epoch{epoch}.pth"
    onnx_path = weight_path.with_suffix(".onnx")

    convert_to_onnx(onnx_path)

    input_filepath = Path("data_floorplan/test2/floorplan_resized.png")
    img = Image.open(input_filepath)

    if img.mode == "RGBA":
        img = img.convert("RGB")

    img = (
        torch.from_numpy(np.array(img).transpose(2, 0, 1) / 255.0).unsqueeze(0).float()
    )
    img = np.array(img)

    # test the onnx model
    ort_session = onnxruntime.InferenceSession(str(onnx_path))
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)
    mask = ort_outs[0][0]

    print(mask.shape)

    save_mask_channel(mask, 0)
    save_mask_channel(mask, 1)

    mask_image = mask_to_image(mask, [0, 1])

    mask_filepath = input_filepath.with_name(f"{input_filepath.stem}_mask.png")
    mask_image.save(mask_filepath)
