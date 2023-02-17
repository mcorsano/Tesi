from torchvision import transforms
from torchvision.utils import save_image
import torch
import numpy as np
import os
from PIL import Image 
from skimage.transform import iradon
import cv2
import tifffile


toTensor = transforms.ToTensor()


images_directory = "./tests/sr/"
sinograms_folder = "./reconstruction/sinograms"
slices_folder = "./reconstruction/slices"

image_width = 1024
image_height = 1024
n_angles = 300


for row in range(image_height):  

    sinogram = torch.zeros(image_width, n_angles, dtype = torch.float64)

    for idx in range(n_angles):
        image_path = os.path.join(images_directory, "sr-denoised_unet_rgb_{:04d}.tif".format(idx))
        sr_image = toTensor(Image.open(image_path))[0]*2.07
        print(image_path)

        # print(torch.amax(sr_image))
        # print(torch.amin(sr_image))
        for el in range(image_width):
            if(row<image_height-1):
                sinogram[el,idx] = sr_image[(image_height-row)-1,el]

    sinogram = sinogram.numpy()
    # print(np.max(sinogram))
    # print(np.min(sinogram))

    # for offset in range(5,7,1):
    #     sinogram_rolled = np.roll(sinogram, offset, 0)
        #sinogram = cv2.resize(sinogram.numpy().astype(np.float64), dsize=(900,1024))
    cv2.imwrite(os.path.join(sinograms_folder, "sinogram_{}.tiff".format(row)), sinogram)

    theta = np.linspace(0., 180., n_angles, endpoint=False)                                 
    reconstruction_fbp = iradon(sinogram, theta=theta, filter_name='shepp-logan')
    print(np.max(reconstruction_fbp))
    print(np.min(reconstruction_fbp))
    cv2.imwrite(os.path.join(slices_folder, "slice_{}.tiff".format(row)), reconstruction_fbp)