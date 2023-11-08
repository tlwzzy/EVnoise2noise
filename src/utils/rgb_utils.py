#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF

import os
import numpy as np
from math import log10
from datetime import datetime
from PIL import Image
import Imath

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def clear_line():
    """Clears line from any characters."""

    print('\r{}'.format(' ' * 80), end='\r')


def progress_bar(batch_idx, num_batches, report_interval, train_loss):
    """Neat progress bar to track training."""

    dec = int(np.ceil(np.log10(num_batches)))
    bar_size = 21 + dec
    progress = (batch_idx % report_interval) / report_interval
    fill = int(progress * bar_size) + 1
    print('\rBatch {:>{dec}d} [{}{}] Train loss: {:>1.5f}'.format(batch_idx + 1, '=' * fill + '>', ' ' * (bar_size - fill), train_loss, dec=str(dec)), end='')


def time_elapsed_since(start):
    """Computes elapsed time since start."""

    timedelta = datetime.now() - start
    string = str(timedelta)[:-7]
    ms = int(timedelta.total_seconds() * 1000)

    return string, ms


def show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr):
    """Formats validation error stats."""

    clear_line()
    print('Train time: {} | Valid time: {} | Valid loss: {:>1.5f} | Avg PSNR: {:.2f} dB'.format(epoch_time, valid_time, valid_loss, valid_psnr))


def show_on_report(batch_idx, num_batches, loss, elapsed):
    """Formats training stats."""

    clear_line()
    dec = int(np.ceil(np.log10(num_batches)))
    print('Batch {:>{dec}d} / {:d} | Avg loss: {:>1.5f} | Avg train time / batch: {:d} ms'.format(batch_idx + 1, num_batches, loss, int(elapsed), dec=dec))


def plot_per_epoch(ckpt_dir, title, measurements, y_label):
    """Plots stats (train/valid loss, avg PSNR, etc.)."""

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(measurements) + 1), measurements)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Epoch')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()

    fname = '{}.png'.format(title.replace(' ', '-').lower())
    plot_fname = os.path.join(ckpt_dir, fname)
    plt.savefig(plot_fname, dpi=200)
    plt.close()


def reinhard_tonemap(tensor):
    """Reinhard et al. (2002) tone mapping."""

    tensor[tensor < 0] = 0
    return torch.pow(tensor / (1 + tensor), 1 / 2.2)


def psnr(input, target):
    """Computes peak signal-to-noise ratio."""
    
    return 10 * torch.log10(1 / F.mse_loss(input, target))


def create_montage(img_name, noise_type, save_path, source_t, denoised_t, clean_t, show):
    """Creates montage for easy comparison."""

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    fig.canvas.manager.set_window_title(img_name.capitalize()[:-4])

    # Bring tensors to CPU
    source_t = source_t.cpu().narrow(0, 0, 3)
    denoised_t = denoised_t.cpu()
    clean_t = clean_t.cpu()

    # optional: change black background to white
    # zero_tensor = torch.zeros_like(source_t, dtype=torch.uint8)
    source_mask = (source_t == 0).all(dim=0)
    denoised_mask = (denoised_t == 0).all(dim=0)
    clean_mask = (clean_t == 0).all(dim=0)

    source_t[:,source_mask] = 255
    denoised_t[:,denoised_mask] = 255
    clean_t[:,clean_mask] = 255

    
    # source = tvF.to_pil_image(source_t)
    source = tvF.to_pil_image(torch.clamp(source_t,0,1))
    denoised = tvF.to_pil_image(torch.clamp(denoised_t, 0, 1))
    # clean = tvF.to_pil_image(clean_t)
    clean = tvF.to_pil_image(torch.clamp(clean_t, 0 ,1))

    # Build image montage
    psnr_vals = [psnr(source_t, clean_t), psnr(denoised_t, clean_t)]
    titles = ['Input: {:.2f} dB'.format(psnr_vals[0]),
              'Denoised: {:.2f} dB'.format(psnr_vals[1]),
              'Ground truth']
    zipped = zip(titles, [source, denoised, clean])
    for j, (title, img) in enumerate(zipped):
        ax[j].imshow(img)
        ax[j].set_title(title)
        ax[j].axis('off')

    # Open pop up window, if requested
    if show > 0:
        plt.show()

    # Save to files
    fname = os.path.splitext(img_name)[0]
    source.save(os.path.join(save_path, f'{fname}-{noise_type}-noisy.png'))
    denoised.save(os.path.join(save_path, f'{fname}-{noise_type}-denoised.png'))
    fig.savefig(os.path.join(save_path, f'{fname}-{noise_type}-montage.png'), bbox_inches='tight')

    return denoised


class AvgMeter(object):
    """Computes and stores the average and current value.
    Useful for tracking averages such as elapsed times, minibatch losses, etc.
    """

    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0.
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mask_images_torch(image_batch, threshold=30):
    """
    Convert a batch of RGB images to grayscale, create a mask where the brightness
    is less than the threshold, and apply the mask to the original images.
    
    Parameters:
    - image_batch: A torch tensor of shape (batch_size, height, width, channels)
    - threshold: The brightness level below which to mask the image
    
    Returns:
    - image_batch_masked: The batch of images with the mask applied
    - mask: The mask that was applied to the image batch
    """
    # Convert the image batch to float for mean calculation
    image_batch_float = image_batch.float()
    
    # Convert to grayscale by taking the mean across the channels (assuming RGB format)
    if image_batch_float.shape[-1] == 3:
        image_gray = torch.mean(image_batch_float, dim=-1, keepdim=True)
    else:
        image_gray = image_batch_float
    
    # Create a mask for each image where the brightness is less than the threshold
    mask = image_gray < threshold
    
    # Expand mask to have the same number of channels as the input
    mask_expanded = mask.expand(-1, -1, -1, 3)
    
    # Apply the mask to the original image to set the pixels to 0 where the mask is True
    image_batch_masked = image_batch_float.clone()
    image_batch_masked[mask_expanded] = 0
    
    # Cast the masked images back to the original data type
    image_batch_masked = image_batch_masked.type(image_batch.dtype)
    
    return image_batch_masked, mask

def mask_image_torch(image, threshold=30):
    """
    Convert an RGB image to grayscale, create a mask where the brightness
    is less than the threshold, and apply the mask to the original image.
    
    Parameters:
    - image: A torch tensor of shape (height, width, channels)
    - threshold: The brightness level below which to mask the image
    
    Returns:
    - image_masked: The image with the mask applied
    - mask: The mask that was applied to the image
    """
    # Convert the image to float for mean calculation
    image = tvF.to_pil_image(image)
    image_float = torch.tensor(np.array(image)).float().permute(2, 0, 1)
    
    # Convert to grayscale by taking the mean across the channels (assuming RGB format)
    if image_float.shape[0] == 3:
        image_gray = torch.mean(image_float, dim=0, keepdim=True)
    else:
        image_gray = image_float
    
    # Create a mask where the brightness is less than the threshold
    mask = image_gray < threshold
    
    # Expand mask to have the same number of channels as the input
    if image_float.shape[0] == 3:
        mask = mask.expand(3, -1, -1)

    # demo: how to use the mask
    # image_masked[mask] = 0

    return mask