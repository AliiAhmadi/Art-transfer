from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, models
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.vgg16(pretrained = True).features

# Freeze gradients because we do not need to update them.
for parameter in model.parameters():
    parameter.requires_grad_(False)

def preprocess_image(img_path, max_size = 400, shape = None):

    image = Image.open(img_path).convert("RGB")
    plt.imshow(image)
    # print(image.size) 640 * 640

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))]) # ImageNet standard AVG. and Variance
    image = transform(image).unsqueeze(0)

    return image

# Load images
main_img = preprocess_image("1.jpg").to(device)
style_img = preprocess_image("2.jpg", shape=main_img.shape[2:]).to(device) # Resize the style image to match the main image.

# Reverse notmalizing
def converter(tensor):
    img = tensor.clone().detach()
    img = img.numpy().squeeze() # Delete the batch dimension
    img = img.transpose(1, 2, 0)
    img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406)) # Undo normalization
    img = img.clip(0, 1)

    return img

model = model.to(device)


def feature_extract(image, model, layers=None):

    if layers is None:
        layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",
            "28": "conv5_1"
            }


    features = {}
    x = image

    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features

# Calculate the gram matrix.
# Gram is a way to describe the style of an image.

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)
    gram = torch.mm(tensor, tensor.t()) # Product of tensor and its transpose to calculate correlation.
    return gram

main_features = feature_extract(main_img, model)
style_features = feature_extract(style_img, model)

style_grams = {layer : gram_matrix(style_features[layer]) for layer in style_features}

t = main_img.clone().requires_grad_(True)
t = t.to(device)

style_weights = {
    "conv1_1": 1.,
    "conv2_1": 0.8,
    "conv3_1": 0.5,
    "conv4_1": 0.3,
    "conv5_1": 0.1
    }

content_weight = 1
style_weight = 1e6

show_every = 500
optimizer = optim.Adam([t], lr= 0.005)
steps = 5000

for step in range(1, steps+1):
    target_features = feature_extract(t, model)
    content_loss = torch.mean((target_features["conv4_2"] - main_features["conv4_2"])**2)

    style_loss = 0

    for layer in style_weights:
        target_feature = target_features[layer]
        _, d, h, w = target_feature.size()

        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)

        style_loss += layer_style_loss /(d*h*w)

    total_loss = (content_weight * content_loss) + (style_weight * style_loss)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % show_every == 0:
        print("Total loss", total_loss.item())
        plt.imshow(converter(t.cpu()))
        plt.show()

# Export the result
plt.imsave("result.jpg", converter(t.cpu()))