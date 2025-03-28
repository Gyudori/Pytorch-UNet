import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
from pathlib import Path

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.utils import get_prediction_debug_image


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, output_dir: Path, step: int):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    idx = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            prediction = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                prediction2 = (F.sigmoid(prediction) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(prediction2, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true2 = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                prediction2 = F.one_hot(prediction.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(prediction2[:, 1:], mask_true2[:, 1:], reduce_batch_first=False)
            
            if output_dir is not None:
                image = image[0].cpu()
                prediction = prediction.float().cpu()
                mask_pred = prediction.argmax(dim=1)[0].float().cpu().unsqueeze(0)
                mask_true = mask_true.cpu()
                # save the debug image
                debug_image = get_prediction_debug_image(image, prediction, mask_pred, mask_true)
                save_image(debug_image, output_dir / f'image_{idx:03d}_step_{step}.png')
                
            idx += 1
                        
            

    net.train()
    return dice_score / max(num_val_batches, 1)
