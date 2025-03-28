from pathlib import Path

from PIL import Image

import tqdm


def main():
    rootDirpath = Path(".")
    inputDirpath = rootDirpath / "pair"
    imgsDirpath = rootDirpath / "imgs"
    masksDirpath = rootDirpath / "masks"

    imgsDirpath.mkdir(parents=True, exist_ok=True)
    masksDirpath.mkdir(parents=True, exist_ok=True)

    filepaths = list(inputDirpath.glob("*.png"))
    filepaths.sort()

    for filepath in tqdm.tqdm(filepaths):        
        filename = filepath.name
        basename = filename.split(".")[0]

        tokens = basename.split("_")
        floorplanId = tokens[4]

        isMask = False

        outputFilename = "us." + floorplanId
        if len(tokens) == 6:
            outputFilename += ".png"
        elif len(tokens) == 7:
            outputFilename += ".gif"
            isMask = True

        inputFilepath = inputDirpath / filename

        if isMask:
            mask = Image.open(inputFilepath)

            # black to white and white to black
            # mask = mask.convert("L")
            # mask = mask.point(lambda x: 255 - x)

            mask.save(masksDirpath / outputFilename)
        else:
            img = Image.open(inputFilepath)
            
            # img to channel 3
            img = img.convert("RGB")
            
            img.save(imgsDirpath / outputFilename)


if __name__ == "__main__":
    main()
