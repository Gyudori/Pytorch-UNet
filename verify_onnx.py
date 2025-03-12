import onnx
import onnxruntime
import numpy as np

from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from utils.data_loading import BasicDataset
from predict import mask_to_image

if __name__ == "__main__":
    run_dir = Path("runs/Feb11_18-23-11_tiso_augmentation_LR_1e-08_BS_1_SCALE_0.5")
    checkpoint_dir = run_dir / "checkpoints"
    onnx_path = checkpoint_dir / "checkpoint_epoch500.onnx"

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(onnx_path)

    test_filepath = Path("data_floorplan/test6/floorplan_resized.png")
    test_image = Image.open(test_filepath)
    if test_image.mode == "RGBA":
        test_image = test_image.convert("RGB")

    enable_resize = False
    if enable_resize:
        max_size = 2048
        ratio = min(max_size / test_image.width, max_size / test_image.height)

        test_image = test_image.resize(
            (int(test_image.width * ratio), int(test_image.height * ratio)),
            resample=Image.Resampling.BICUBIC,
        )

        resized_filepath = Path(test_filepath).parent / (
            Path(test_filepath).stem + "_resized.png"
        )
        test_image.save(resized_filepath)

        print("Resized image size:", test_image.size)

    input = BasicDataset.preprocess(
        None,
        pil_img=test_image,
        scale=1.0,
        is_mask=False,
    )
    input = input.astype(np.float32)
    input = np.expand_dims(input, axis=0)

    print("input shape:", input.shape)

    ort_inputs = {ort_session.get_inputs()[0].name: input}
    ort_outs = ort_session.run(None, ort_inputs)

    print(ort_outs[0].shape)

    mask = ort_outs[0][0]  # (1, 2, 3072, 4096)

    mask = np.argmax(mask, axis=0)  # (3072, 4096)

    out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)

    for i, v in enumerate([0, 1]):
        out[mask == i] = v

    output_image = Image.fromarray(out)
    output_filepath = Path(test_filepath).parent / (
        Path(test_filepath).stem + "_output.jpg"
    )
    output_image.save(output_filepath, format="JPEG", quality=95)
