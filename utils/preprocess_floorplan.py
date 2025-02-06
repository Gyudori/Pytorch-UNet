from pathlib import Path
from PIL import Image
import numpy as np


def process_bim_drawing_file(file: Path, output_dir: Path):
    output_filepath = output_dir / file.name.replace("_bim_drawing.png", ".gif")

    if output_filepath.exists():
        print("Skip existing file", output_filepath)
        return

    image = Image.open(file)

    if image.mode != "RGBA":
        print("Skip non-RGBA image", file)
        return

    rgba_array = np.array(image)
    alpha_channel = rgba_array[:, :, 3]

    rgba_array[:, :, :3] = np.where(
        alpha_channel[:, :, None] > 0, [255, 255, 255], rgba_array[:, :, :3]
    )

    bw_image = Image.fromarray(
        (rgba_array[:, :, :3] == [255, 255, 255]).all(axis=-1).astype(np.uint8) * 255
    )

    bw_image.save(
        output_filepath,
        format="GIF",
        append_images=[bw_image],
        save_all=True,
        duration=500,
        loop=0,
    )


def process_floorplan_file(file: Path, output_dir: Path):
    output_filepath = output_dir / file.name.replace("_floorplan.png", ".jpg")

    if output_filepath.exists():
        print("Skip existing file", output_filepath)
        return

    image = Image.open(file)
    rgb_image = image.convert("RGB")

    rgb_image.save(output_filepath, format="JPEG", quality=95)


def main():
    data_root = Path("data_floorplan")
    input_dir = data_root / "collected_priormaps"

    mask_dir = data_root / "masks"
    mask_dir.mkdir(exist_ok=True)

    imgs_dir = data_root / "imgs"
    imgs_dir.mkdir(exist_ok=True)

    for file in input_dir.iterdir():
        if file.name.endswith("_bim_drawing.png"):
            process_bim_drawing_file(file, mask_dir)
        elif file.name.endswith("_floorplan.png"):
            process_floorplan_file(file, imgs_dir)


if __name__ == "__main__":
    main()
