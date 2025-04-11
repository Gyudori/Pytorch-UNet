import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from torchvision.utils import make_grid


def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title("Input image")
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f"Mask (class {i + 1})")
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()


def get_prediction_debug_image(
    image: torch.Tensor,
    prediction: torch.Tensor,
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor = None,
):
    confidence_map = F.softmax(prediction, dim=1)[0].max(dim=0)[0]
    confidence_map = (confidence_map - confidence_map.min()) / (
        confidence_map.max() - confidence_map.min()
    )
    colored_confidence_map = plt.cm.viridis(confidence_map.cpu().detach().numpy())[
        :, :, :3
    ]
    colored_confidence_map = torch.from_numpy(colored_confidence_map).permute(2, 0, 1)

    if true_mask is not None:
        combined = make_grid(
            [
                image,
                colored_confidence_map,
                pred_mask.repeat(3, 1, 1),
                true_mask.repeat(3, 1, 1),
            ],
            nrow=4,
        )
    else:
        combined = make_grid(
            [
                image,
                colored_confidence_map,
                pred_mask.repeat(3, 1, 1),
            ],
            nrow=3,
        )

    return combined


def predict_and_get_debug_image(model, batch, device, amp):
    images, true_mask = batch["image"], batch["mask"]

    image = images.to(
        device=device, dtype=torch.float32, memory_format=torch.channels_last
    )

    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        prediction = model(image)

        image = image[0].cpu()
        prediction = prediction.float().cpu()
        pred_mask = prediction.argmax(dim=1)
        true_mask = true_mask.float().cpu()

        return get_prediction_debug_image(image, prediction, pred_mask, true_mask)
