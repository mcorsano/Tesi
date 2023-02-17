from models import GeneratorResNet
from datasets import denormalize, mean, std
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import argparse
import os
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.utils import save_image, make_grid
from PIL import Image

        
parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default='../data/denoised_unet_rgb/', help="Path to image")
parser.add_argument("--checkpoint_model", type=str, default='./saved_models/generator_150.pth', help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
opt = parser.parse_args()
print(opt)

os.makedirs("images/outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and load model checkpoint
generator = GeneratorResNet(in_channels=3).to(device)
generator.load_state_dict(torch.load(opt.checkpoint_model, map_location=device))
generator.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

for image in os.listdir(opt.image_folder):
            
    image_path = os.path.join(opt.image_folder, image)
    print(image_path)

    # Prepare input
    pil_image = Image.open(image_path)
    image_tensor = Variable(transform(pil_image)).to(device).unsqueeze(0)

    image_tensor = torch.squeeze(image_tensor, 0)
    image_tensor = image_tensor

    image_tensor = image_tensor.to(device)
    image_tensor = image_tensor.unsqueeze(0)

    # Upsample image
    with torch.no_grad():
        sr_image = denormalize(generator(image_tensor)).cpu()


    # Save image
    #sr_image = transforms.Grayscale(num_output_channels=1)(sr_image)
    lr_image = nn.functional.interpolate(image_tensor, scale_factor=4)
    sr_image = transforms.Grayscale(num_output_channels=1)(sr_image)
    save_image(sr_image, f"./tests/sr/sr-{image}", normalize=True)
    save_image(lr_image, f"./tests/interpolated/interp-{image}", normalize=True)