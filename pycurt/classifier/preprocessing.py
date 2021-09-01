from medpy.filter import otsu
from skimage import morphology
from scipy import ndimage
from batchgenerators.utilities.file_and_folder_operations import *
import math
import nibabel as nib
from skimage import exposure
import numpy as np


def remove_noise(image, modality='ct'):

    # morphology.dilation creates a segmentation of the image
    # If one pixel is between the origin and the edge of a square of size
    # 5x5, the pixel belongs to the same class
    
    # We can instead use a circule using: morphology.disk(2)
    # In this case the pixel belongs to the same class if it's between the origin
    # and the radius

    image_np_s, _ = crop_background3D(image, modality=modality)
    segmentation = morphology.dilation(image_np_s, np.ones((5, 5, 5)))
    labels, _ = ndimage.label(segmentation)
    
    label_count = np.bincount(labels.ravel().astype(np.int64))
    # The size of label_count is the number of classes/segmentations found
    
    # We don't use the first class since it's the background
    label_count[0] = 0
    
    # We create a mask with the class with more pixels
    # In this case should be the brain
    mask = labels == label_count.argmax()
   
    # Improve the brain mask
    mask = morphology.dilation(mask, np.ones((5, 5, 5)))
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3, 3)))

    image[mask!=True]= 0

    return image, mask


def smallest(num1, num2, num3):

    if (num1 < num2) and (num1 < num3):
        smallest_num = num1
    elif (num2 < num1) and (num2 < num3):
        smallest_num = num2
    else:
        smallest_num = num3

    return smallest_num


def extract_middleSlice(image, scan):

    x, y, z = image.shape
    s = smallest(x, y, z)
    if (s == z and scan == 3) or scan == 2:
        ms = math.ceil(image.shape[2]/2)-1
        return image[:, :, ms].astype('float32')
    elif (s == y and scan == 3) or scan == 1:
        ms = math.ceil(image.shape[1]/2)-1
        return image[:, ms, :].astype('float32')
    else:
        ms = math.ceil(image.shape[0]/2)-1
        return image[ms, :, :].astype('float32')


def crop_background3D(image_np, modality='ct'):

    if modality == 'mr':
        image_np_s = exposure.equalize_hist(image_np)
        image_np_s = ndimage.gaussian_filter(image_np_s, sigma=(3, 3, 3), order=0)
    else:
        image_np_s = image_np
    ms = extract_middleSlice(image_np_s, 3)
    threshold = otsu(ms)
    output_data = image_np_s > threshold
    output_data = output_data.astype(int)
    img = image_np*output_data
    if modality == 'ct':
        x, y, z = np.where(img>0)
        mask = np.zeros((image_np.shape))
        if image_np.shape[2] > 1:
            mask[x.min():x.max(), y.min():y.max(), z.min():z.max()] = 1
        else:
            mask[x.min():x.max(), y.min():y.max(), :] = 1
        img = image_np*mask
    else:
        mask = output_data
# 
    return img, mask


def run_preprocessing(case, modality='ct'):
    
    imgs_nib = [nib.load(i) for i in case]
    imgs_nib = [nib.as_closest_canonical(i) for i in imgs_nib]
    imgs_npy = [i.get_fdata() for i in imgs_nib]
    
    tmp = []
    for im in imgs_npy:
        if len(im.shape) == 4:
            tmp.append(im[:, :, :, 0])
        elif len(im.shape) == 3 and not [x for x in im.shape if x == 1]:
            tmp.append(im)
        elif len(im.shape) > 4:
            print('Image has {} dimensions'.format(len(im.shape)))
    if not tmp:
        return None
    else:
        imgs_npy = tmp

    if modality == 'ct':
        crp = [crop_background3D(i, modality='ct') for i in imgs_npy]
    else:
        crp = [remove_noise(i, modality=modality) for i in imgs_npy]
    imgs_npy = [x[0] for x in crp]
    nonzero_masks = [x[1] for x in crp]
    imgs_npy = np.concatenate([i[None] for i in imgs_npy]).astype(np.float32)

    # now find the nonzero region and crop to that
    nonzero = [np.array(np.where(i > 0)) for i in nonzero_masks]
    nonzero = [[np.min(i, 1), np.max(i, 1)] for i in nonzero]
    nonzero = np.array([np.min([i[0] for i in nonzero], 0), np.max([i[1] for i in nonzero], 0)]).T
    # nonzero now has shape 3, 2. It contains the (min, max) coordinate of nonzero voxels for each axis

    # now crop to nonzero
    imgs_npy = imgs_npy[:,
               nonzero[0, 0] : nonzero[0, 1] + 1,
               nonzero[1, 0]: nonzero[1, 1] + 1,
               nonzero[2, 0]: nonzero[2, 1] + 1,
               ]

    # now we create a brain mask that we use for normalization
    nonzero_masks = [i != 0 for i in imgs_npy]
    brain_mask = np.zeros(imgs_npy.shape[1:], dtype=bool)
    for i in range(len(nonzero_masks)):
        brain_mask = brain_mask | nonzero_masks[i]

    for i in range(len(imgs_npy)):
        imgs_npy[i][brain_mask == 0] = 0
        if modality == 'ct':
            l = np.percentile(imgs_npy[i][brain_mask > 0], 0.5)
            h = np.percentile(imgs_npy[i][brain_mask > 0], 99.5)
            if l != h:
                image = np.clip(imgs_npy[i], l, h)
            else:
                image = imgs_npy[i]
            imgs_npy[i] = image * 100
            imgs_npy[i][brain_mask == 0] = 0
        elif modality == 'mr':
            mean = imgs_npy[i][brain_mask].mean()
            std = imgs_npy[i][brain_mask].std()
            imgs_npy[i] = (imgs_npy[i] - mean) / (std + 1e-8)
 
    return imgs_npy
