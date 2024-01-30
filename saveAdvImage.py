from PIL import Image
from torchvision.transforms import ToPILImage ,ToTensor
import torch 

def adv_img_save(input_img: Image.Image, modified_tensor:torch.Tensor, output_img_name:str)->torch.Tensor:
    _ , height, width = input_img.shape

    input_tensor_resized = torch.nn.functional.interpolate(
        modified_tensor.unsqueeze(0),
        size=(height, width),
        mode="bicubic"
    ).squeeze(0)


    denormalized_tensor = input_tensor_resized * 255


    to_pil = ToPILImage()
    original_image = to_pil(denormalized_tensor.byte())

    modified_image = original_image
    modified_image.save(f"image/{output_img_name}.jpg")

    to_tensor = ToTensor()
    modified_tensor = to_tensor(modified_image)
    output_tensor = (modified_tensor  * 255).to(torch.uint8)

    return output_tensor , input_tensor_resized