import torch
import torch.onnx

from unet import UNet

from pathlib import Path


def convert_to_onnx(model, input_size, onnx_path):
    # 모델을 평가 모드로 설정
    model.eval()

    # 더미 입력 생성 (배치 크기 1)
    dummy_input = torch.randn(1, *input_size)

    # ONNX 내보내기
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


if __name__ == "__main__":
    model = UNet(n_channels=3, n_classes=2)

    run_dir = Path("runs/Feb12_17-56-30_tiso_augmentation_LR_1e-08_BS_1_SCALE_1.0")
    epoch = 200
    weight_path = run_dir / f"checkpoints/checkpoint_epoch{epoch}.pth"

    state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
    del state_dict["mask_values"]
    model.load_state_dict(state_dict)

    input_size = (3, 4096, 4096)
    onnx_path = weight_path.with_suffix(".onnx")
    convert_to_onnx(model, input_size, onnx_path)
