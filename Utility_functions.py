import numpy as np
from PIL import Image
import torch
import os
from torchvision import transforms

def has_file_allowed_extension(filename, extensions):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.JPG', '.PNG']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()
        return img.convert('RGB')

class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"], self.model(x)['logVar']

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask).float()
        self.mask = self.mask.to('cpu')

    def __call__(self, model_output):
        print('Segment output', (model_output[:, :, :] * self.mask).sum())
        return (torch.relu(model_output[:, :, :]) * self.mask).sum()
