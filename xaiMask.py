
from matplotlib import colormaps as cm
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch 
from scipy.ndimage import gaussian_filter


def xai_mask(activation_map: Image.Image,input_tensor: Image.Image,sensitivity: float) -> torch.Tensor:

    cmap = cm.get_cmap("jet")
    mask = to_pil_image(activation_map[0].squeeze(0), mode='F')
    mask1 = mask.resize(to_pil_image(input_tensor).size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(mask1) ** 2)[:, :, :3]).astype(np.uint8)



    num_colors = overlay.shape[0]

    cmap_reversed = plt.get_cmap("jet")

    gradient_colors_reversed = (cmap_reversed(np.linspace(1, sensitivity, num_colors)) * 255).astype(np.uint8)


    indices = []
    for color in gradient_colors_reversed:
        match_indices = np.where(np.all(overlay[..., :3] == color[:3], axis=-1))
        indices.append(match_indices)


    indices = (np.concatenate([ind for ind in indices], axis=1)).astype(int)
    modified_tensor = input_tensor.clone().detach().requires_grad_(True)


    max_index_0 = modified_tensor.shape[1]
    max_index_1 = modified_tensor.shape[2]
    indices = (np.clip(indices[0], 0, max_index_0 - 1), np.clip(indices[1], 0, max_index_1 - 1))

    new_mask = torch.zeros_like(modified_tensor)

    new_mask[:, indices[0], indices[1]] = 1


    def gaussian_smooth_mask(mask, sigma=1.0):

        mask_np = mask.numpy()

        mask_smoothed = np.zeros_like(mask_np)
        for channel in range(mask_np.shape[0]):
            mask_smoothed[channel] = gaussian_filter(mask_np[channel], sigma=sigma)

        return torch.from_numpy(mask_smoothed)

    modified_tensor_smoothed = gaussian_smooth_mask(new_mask, sigma=2.0)

    return modified_tensor_smoothed ,modified_tensor