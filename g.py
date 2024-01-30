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


model = InceptionResnetV1(pretrained='vggface2',classify=True).eval()
img = read_image("image/m.jpg")
# model = resnet18(weights='IMAGENET1K_V1').eval()

# input_tensor = normalize(resize(img, (224, 224), antialias=False) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


input_tensor = resize(img, (224, 224), antialias=False) / 255

with torch.no_grad():
    out = model(input_tensor.unsqueeze(0))

probs = torch.nn.functional.softmax(out, dim=1)
predicted_class_index = torch.argmax(probs).item()

print(" Before purturb ",predicted_class_index)




mtcnn = MTCNN(keep_all=True)

aligned, prob = mtcnn(to_pil_image(img), return_prob=True)

if aligned is not None:
    print('1 Face detected with probability: {:8f}'.format(prob.item()))
else:
    print('1 No face detected in the image.')


with GradCAMpp(model,'block8') as cam_extractor:
    output = model(input_tensor.unsqueeze(0))
    class_index = output.squeeze(0).argmax().item()
    activation_map = cam_extractor(class_index, output)



cmap = cm.get_cmap("jet")
mask = to_pil_image(activation_map[0].squeeze(0), mode='F')
mask1 = mask.resize(to_pil_image(input_tensor).size, resample=Image.BICUBIC)
overlay = (255 * cmap(np.asarray(mask1) ** 2)[:, :, :3]).astype(np.uint8)

# print(overlay)


values = np.linspace(0, 1, 256)


# Get the "jet" colormap
# cmap_jet = plt.get_cmap("jet_r")

# # Create a plot to visualize the colormap
# fig, ax = plt.subplots(figsize=(6, 1))
# cax = ax.imshow(np.vstack((values, values)), aspect='auto', cmap=cmap_jet)
# ax.set_xticks([])
# ax.set_yticks([0, 1])
# ax.set_yticklabels([0, 1])
# fig.colorbar(cax, orientation='horizontal')

# plt.title("Color Range of 'jet' Colormap")
# plt.show()

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

num_colors = overlay.shape[0]

# cmap_reversed = plt.get_cmap("jet_r")
cmap_reversed = plt.get_cmap("jet")

# Generate a linear gradient of colors from high to low sensitivity
gradient_colors_reversed = (cmap_reversed(np.linspace(1, 0.8, num_colors)) * 255).astype(np.uint8)


indices = []
for color in gradient_colors_reversed:
    match_indices = np.where(np.all(overlay[..., :3] == color[:3], axis=-1))
    indices.append(match_indices)


indices = (np.concatenate([ind for ind in indices], axis=1)).astype(int)



####################################################################################
####################################################################################
#less disortion and invinsibilty

# original_array = np.array(img)
# blend_weight = 0.8


# modified_array = original_array.copy()

# modified_pixels = modified_array[:, indices[0], indices[1]]

# print(modified_pixels)

# print( blend_weight * original_array[:, indices[0], indices[1]])

# modified_pixels = (
#     blend_weight * original_array[:, indices[0], indices[1]]
#     + (1 - blend_weight) * modified_pixels
#     + 0.8  # You can adjust this value to make the change more visible
# )

# print(modified_pixels)

# modified_array[:, indices[0], indices[1]] = modified_pixels *0

# # Save the modified image
# modified_image = Image.fromarray(modified_array.transpose(1, 2, 0))
# modified_image.save("image/new.jpg")
# to_tensor = ToTensor()
# modified_tensor = to_tensor(modified_image)
# output_tensor = (modified_tensor  * 255).to(torch.uint8)


#########################################################################################
#########################################################################################


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

criterion = nn.CrossEntropyLoss()


learning_rate = 0.01
num_iterations = 1


for i in range(num_iterations):
    output3 = model(modified_tensor.unsqueeze(0))

    target_class = torch.tensor([0]) 

    loss = criterion(output3, target_class)

    gradient = torch.autograd.grad(loss, modified_tensor)[0]
    
    # FGSM perturbation
    modified_tensor = modified_tensor + learning_rate *torch.sign(gradient)*modified_tensor_smoothed

    # PGD perturbation
    # perturbation = learning_rate * torch.sign(gradient) * modified_tensor_smoothed
    # modified_tensor = torch.clamp(modified_tensor + perturbation, 0, 1)
    # modified_tensor = torch.clamp(modified_tensor, 0, 1)

    # modified_tensor =modified_tensor*modified_tensor_smoothed

with torch.no_grad():
    out = model(modified_tensor.unsqueeze(0))

probs = torch.nn.functional.softmax(out, dim=1)
predicted_class_index = torch.argmax(probs).item()

print(" After Purturb ",predicted_class_index)

####################################################################################
####################################################################################
# modified tensor convet to original img size

# normalize_mean = [0.485, 0.456, 0.406]
# normalize_std = [0.229, 0.224, 0.225]

# denormalize = transforms.Compose([
#     transforms.Normalize(mean=[0, 0, 0], std=[1 / m for m in normalize_std]),
#     transforms.Normalize(mean=[-m for m in normalize_mean], std=[1, 1, 1]),
# ])

_ , height, width = img.shape

# resize_inverse = transforms.Compose([
#     transforms.Resize((height, width)),
# ])
# denormalized_tensor = denormalize(modified_tensor)


# denormalized_image = transforms.ToPILImage()(denormalized_tensor)

# original_image = resize_inverse(denormalized_image)
    
input_tensor_resized = torch.nn.functional.interpolate(
    modified_tensor.unsqueeze(0),
    size=(height, width),
    mode="bicubic"
).squeeze(0)


denormalized_tensor = input_tensor_resized * 255

# Convert the tensor to a PIL Image
to_pil = ToPILImage()
original_image = to_pil(denormalized_tensor.byte())




# Save the modified image
modified_image = original_image
modified_image.save("image/new1.jpg")
to_tensor = ToTensor()
modified_tensor = to_tensor(modified_image)
output_tensor = (modified_tensor  * 255).to(torch.uint8)

#########################################################################################
#########################################################################################



original_gradcam = overlay_mask(to_pil_image(input_tensor_resized), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
# attack_gradcam = overlay_mask(to_pil_image(adImg), to_pil_image(tensor, mode='F'), alpha=0.5)
# attack_gradcam = overlay_mask(noise, to_pil_image(inputShapeNewamapTensor), alpha=0.5)

#################################################################
################################################################
################################################################

# print(output_tensor)
# print(img)


new_input_tensor = normalize(resize(output_tensor, (224, 224), antialias=False) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

mtcnn = MTCNN(keep_all=True)

aligned, prob = mtcnn(to_pil_image(modified_tensor), return_prob=True)

if aligned is not None:
    print('2 Face detected with probability: {:8f}'.format(prob.item()))
else:
    print('2 No face detected in the image.')


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
plt.imshow(activation_map[0].squeeze(0).numpy())
plt.title("activation map")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(new_attack_gradcam) 
# plt.imshow(overlay)
# plt.imshow(modified_image)
plt.title("attack image")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(new_activation_map[0].squeeze(0).numpy())
plt.title("attack actiavtion map")
plt.axis('off')

plt.tight_layout()
plt.show()