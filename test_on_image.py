from models import GeneratorRRDB
from datasets import denormalize, mean, std
import torch
from torch.autograd import Variable
import argparse
import os
import tifffile
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default='../../data/denoisedImages/denoisedImages_unet/', help="Path to image")
parser.add_argument("--checkpoint_model", type=str, default='./saved_models/generator_500.pth', help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
opt = parser.parse_args()
print(opt)

os.makedirs("images/outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for image in os.listdir(opt.image_folder):
    
    #if not os.path.exists(f"images/outputs/sr-{image}") :
    image_path = os.path.join(opt.image_folder, image)
    print(image_path)

    # Define model and load model checkpoint
    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
    generator.load_state_dict(torch.load(opt.checkpoint_model, map_location=torch.device(device)))
    generator.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    toPIL = transforms.ToPILImage()

    # Prepare input
    img = Image.open(image_path)
    img = np.asarray(img)/2.062036

    img = np.repeat(img[..., np.newaxis], 3, -1)
    img = torch.permute(torch.from_numpy(img), (2,0,1))

    save_image(img, f"images/outputs/sr-{image}")



    # image_tensor = Variable(transform(toPIL(img))).to(device).unsqueeze(0)
    # image_tensor = Variable(transform(Image.open(image_path))).to(device).unsqueeze(0)
    image_tensor = Variable(img).to(device).unsqueeze(0)
    
    # print(torch.max(image_tensor))
    # print(torch.min(image_tensor))

    # Upsample image
    with torch.no_grad():
        sr_image = denormalize(generator(image_tensor)).cpu()
        # sr_image = generator(image_tensor).cpu()

    print(torch.max(sr_image))
    print(torch.min(sr_image))

    # Save image
    # sr_image = transforms.Grayscale(num_output_channels=1)(sr_image)
    # sr_image = sr_image/torch.max(sr_image)
    # print(torch.max(sr_image))
    # print(torch.min(sr_image))

    # save_image(sr_image[0], f"images/outputs/sr-{image}")
    # tifffile.imsave("./images/outputs/sr-{image}", sr_image)