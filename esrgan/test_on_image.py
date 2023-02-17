from models import GeneratorRRDB
from datasets import denormalize, mean, std
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default='../data/mouse_noisy/', help="Path to image")
parser.add_argument("--checkpoint_model", type=str, default='./saved_models/generator_17.pth', help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
opt = parser.parse_args()
print(opt)

#os.makedirs("images/outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for image in os.listdir(opt.image_folder):

    image_path = os.path.join(opt.image_folder, image)
    print(image_path)

    # Define model and load model checkpoint
    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
    generator.load_state_dict(torch.load(opt.checkpoint_model, map_location=torch.device(device)))
    generator.eval()

    transform = transforms.Compose([transforms.Resize((63,434), Image.BICUBIC),transforms.ToTensor(), transforms.Normalize(mean, std)])

    # Prepare input
    image_tensor = Variable(transform(Image.open(image_path))).to(device).unsqueeze(0)

    # Upsample image
    with torch.no_grad():
        sr_image = denormalize(generator(image_tensor)).cpu()

    # Save image
    # fn = opt.image_folder.split("/")[-1]
    sr_image = transforms.Grayscale(num_output_channels=1)(sr_image)
    save_image(sr_image, f"images/outputs/sr-{image}")

    img = Image.open(image_path)
    img_lr_interp = img.resize((1736,252), Image.Resampling.BICUBIC)
    img_lr_interp.save(f"images/outputs/interp-{image}")
