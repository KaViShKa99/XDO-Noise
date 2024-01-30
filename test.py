from torchvision.models import resnet18
from torchvision.transforms.functional import normalize, resize,to_pil_image,to_tensor
from torchvision.io.image import read_image
import matplotlib.pyplot as plt
from torchcam.methods import SmoothGradCAMpp,GradCAMpp
from torchcam.utils import overlay_mask
import torch 
from facenet_pytorch import InceptionResnetV1,MTCNN
from torchvision.transforms import ToPILImage ,ToTensor
import torch.nn.functional as F
import numpy as np
from PIL import Image
from matplotlib import colormaps as cm
from torch import nn
from torchvision import transforms
from skimage import measure
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from torchvision.models import resnet18


from xaiMask import xai_mask
from xdoNoise import xdo_noise
from saveAdvImage import adv_img_save

#########################################################################################
#########################################################################################
img = read_image("image/m.jpg")
input_tensor = resize(img, (224, 224), antialias=False) / 255




model = InceptionResnetV1(pretrained='vggface2',classify=True).eval()

# img = read_image("image/m.jpg")
# input_tensor = resize(img, (224, 224), antialias=False) / 255



with torch.no_grad():
    out = model(input_tensor.unsqueeze(0))

    probs = torch.nn.functional.softmax(out, dim=1)
    predicted_class_index = torch.argmax(probs).item()
    
    

print(predicted_class_index)


with GradCAMpp(model,'block8') as cam_extractor:
    output = model(input_tensor.unsqueeze(0))
    class_index = output.squeeze(0).argmax().item()
    activation_map = cam_extractor(class_index, output)


modified_tensor_smoothed , modified_tensor= xai_mask(activation_map,input_tensor,0.5)    


learning_rate = 0.01
num_iterations = 1


for i in range(num_iterations):

    masked_gradient, full_gradient = xdo_noise (model,'block8', modified_tensor)
    # FGSM perturbation

    modified_tensor = modified_tensor + learning_rate *torch.sign(masked_gradient)
    # PGD perturbation

    # perturbation = learning_rate * torch.sign(masked_gradient) 
    # modified_tensor = torch.clamp(modified_tensor + perturbation, 0, 1)
    # modified_tensor = torch.clamp(modified_tensor, 0, 1)


with torch.no_grad():
    out = model(modified_tensor.unsqueeze(0))

    probs1 = torch.nn.functional.softmax(out, dim=1)
    predicted_class_index1 = torch.argmax(probs1).item()


print(predicted_class_index1)


output_tensor,input_tensor_resized = adv_img_save(img,modified_tensor,"new_adv")


#########################################################################################
#########################################################################################


original_gradcam = overlay_mask(to_pil_image(input_tensor_resized), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)



new_input_tensor = normalize(resize(output_tensor, (224, 224), antialias=False) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


with GradCAMpp(model,'block8') as cam_extractor:
    output1 = model(new_input_tensor.unsqueeze(0))
    class_index1 = output1.squeeze(0).argmax().item()
    new_activation_map = cam_extractor(class_index1, output1)

new_attack_gradcam = overlay_mask(to_pil_image(output_tensor), to_pil_image(new_activation_map[0].squeeze(0), mode='F'), alpha=0.5)
#################################################################
#################################################################
#################################################################

plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
# plt.imshow(to_pil_image(img))
plt.imshow(original_gradcam)
plt.title("original image")
plt.axis('off')

plt.subplot(2, 2, 2)
# plt.imshow(activation_map[0].squeeze(0).numpy())
# plt.title("activation map")

plt.imshow(full_gradient[0] * 0.5 + 0.5)
plt.title("full gradient")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(new_attack_gradcam) 
# plt.imshow(overlay)
# plt.imshow(modified_image)
plt.title("attack image")
plt.axis('off')

plt.subplot(2, 2, 4)
# plt.imshow(new_activation_map[0].squeeze(0).numpy())
# plt.title("attack actiavtion map")


plt.imshow(masked_gradient[0] * 0.5 + 0.5)
plt.title("masked gradient")
plt.axis('off')

# plt.imshow(gradient[0] * 0.5 + 0.5)
# plt.imshow(masked_gradient[0] * 0.5 + 0.5)

plt.tight_layout()
plt.show()