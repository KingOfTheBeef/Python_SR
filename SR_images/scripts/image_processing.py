import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from annoy import AnnoyIndex
import random
import time
from skimage import data, io, filters, color, transform, util
from skimage.transform import pyramid_gaussian
from scipy import ndimage

def read_image(filename):
  return util.img_as_float(io.imread(filename))

def show_image(img):
    plt.imshow(img)
    plt.show()


def gaussian_filter(img, kernel_size=(5,5), std=1):
    return filters.gaussian(img)#, kernel_size, std, std)

def RGB2YIQ(img):
    return color.rgb2yiq(img)

def YIQ2RGB(img):
    return color.yiq2rgb(img)

def get_high_and_low_frequency(img, sigma=5):
  #low = ndimage.gaussian_filter(img, (5, 5, 1))
  low = filters.gaussian(
    img, sigma=(sigma, sigma), multichannel=True)
  high = img - low
  return high, low

def get_patch_components(image, patches, blur_sigma=3, gaussian_sigma=3):
  # img_raw = io.imread(filename)
  # img_downsize = transform.rescale(img_raw, scale_factor, multichannel=True)
  image_blurred = filters.gaussian(image, sigma=(blur_sigma, blur_sigma), multichannel=True)

  A_images = []
  for i in range(len(patches)):
    A_images.append([
                     image_blurred[patches[i, 0, 0]:patches[i, 0, 1],patches[i, 1, 0]:patches[i, 1, 1]],
                     image[patches[i, 0, 0]:patches[i, 0, 1],patches[i, 1, 0]:patches[i, 1, 1]]
                          #RGB2YIQ(image_blurred[patches[i, 0, 0]:patches[i, 0, 1],patches[i, 1, 0]:patches[i, 1, 1]]),
                          #RGB2YIQ(image[patches[i, 0, 0]:patches[i, 0, 1],patches[i, 1, 0]:patches[i, 1, 1]])
    ])
  #B_image = image_blurred #RGB2YIQ(image_blurred)
  

  A_images_high = []
  A_images_low = []

  for i in range(len(A_images)):
    A_images_high.append([])
    A_images_low.append([])
    for j in range(len(A_images[i])):
      high, low = get_high_and_low_frequency(A_images[i][j], sigma=gaussian_sigma)
      # A_images_high[i].append(high)
      # A_images_low[i].append(low)
      A_images_high[i].append(RGB2YIQ(high))
      A_images_low[i].append(RGB2YIQ(low))

  #B_image_high, B_image_low = get_high_and_low_frequency(B_image)

  return A_images_high, A_images_low #, B_image_high, B_image_low

def get_image_components(image, blur_sigma=3, gaussian_sigma=3):
  image_blurred = filters.gaussian(image, sigma=(blur_sigma, blur_sigma), multichannel=True)
  B_image_high, B_image_low = get_high_and_low_frequency(image_blurred, sigma=gaussian_sigma)
  B_image_high = RGB2YIQ(B_image_high)
  B_image_low = RGB2YIQ(B_image_low)
  return B_image_high, B_image_low

def adjust_image_range(image):
  return (image - np.min(image)) / (np.max(image) - np.min(image))


def downsize_img(image, scale_factor=1/4):
  return transform.rescale(image, scale_factor, multichannel=True)

def view_blurs(image, title_prefix=""):
  for i in [0.2, 0.5, 1, 3, 5, 10, 30, 50, 100, 300]:
    high_freq, low_freq = get_high_and_low_frequency(image, sigma=i)  # filters.gaussian(skin_patch, sigma=(i, i), multichannel=True)
    #high_freq = skin_patch - skin_patch_blur
    # print(np.max(high_freq))
    # print(np.min(high_freq))
    # io.imshow(skin_patch_blur)
    # plt.title("Sigma: " + str(i))
    # io.show()
    # print(high_freq.dtype)
    adjusted_high_freq = (high_freq - np.min(high_freq)) / (np.max(high_freq) - np.min(high_freq))
    io.imshow(adjusted_high_freq)
    plt.title(IMAGEPATH + title_prefix + "_Sigma: " + str(i))
    io.show()
    #io.imsave(IMAGEPATH + title_prefix + "_high_frequency_sigma_" + str(i) + ".png", adjusted_high_freq)
