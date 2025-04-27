from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, models
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.vgg16(pretrained = True)
