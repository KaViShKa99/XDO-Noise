from torchcam.methods import GradCAMpp
from torch import nn
import torch 
from PIL import Image

from xaiMask import xai_mask


def xdo_noise (model,last_layer : str,input_tensor: Image.Image ) -> torch.Tensor:

    with GradCAMpp(model,last_layer) as cam_extractor:
        output = model(input_tensor.unsqueeze(0))
        class_index = output.squeeze(0).argmax().item()
        activation_map = cam_extractor(class_index, output)


    modified_tensor_smoothed ,modified_tensor = xai_mask(activation_map,input_tensor,0.5)    

    criterion = nn.CrossEntropyLoss()



    output = model(input_tensor.unsqueeze(0))

    target_class = torch.tensor([0]) 
    loss = criterion(output, target_class)


    gradient = torch.autograd.grad(loss, input_tensor)[0]
    masked_gradient = gradient * modified_tensor_smoothed

    return masked_gradient , gradient