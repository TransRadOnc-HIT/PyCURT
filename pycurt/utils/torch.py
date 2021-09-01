import numpy as np
import torch
import cv2
from pycurt.utils.filemanip import extract_middleSlice
from torch.utils.data import Dataset
import nibabel as nib
import os
import torchio as tio
from medpy.filter import otsu

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if sample is not None:
            image, label,spacing,fn = sample['image'], sample['label'], sample['spacing'], sample['fn']
            # torch.from_numpy(np.float32(image))
            return {'image': torch.from_numpy(np.float32(image)).unsqueeze(dim=0),
                    'label': label,
                    'spacing':spacing,
                    'fn':fn}

   
class ZscoreNormalization(object):
    """ put data in range of 0 to 1 """
   
    def __call__(self,sample):
        
        if sample is not None:
            image, label, spacing, fn = sample['image'], sample['label'], sample['spacing'], sample['fn']
            image -= image.mean()
            if image.std() != 0:
                image /= image.std() 
                
            return {'image': image, 'label': label, 'spacing':spacing, 'fn':fn}
        else:
            return None


class resize_2Dimage:
    """ Args: img_px_size slices resolution(cubic)
              slice_nr Nr of slices """
    
    def __init__(self,img_px_size):
        self.img_px_size=img_px_size
        
    def __call__(self,sample):
        image, label, spacing, fn = sample['image'], sample['label'], sample['spacing'], sample['fn']
        image_n= cv2.resize(image, (self.img_px_size, self.img_px_size), interpolation=cv2.INTER_CUBIC)
        return {'image': image_n,'label': label, 'spacing':spacing, 'fn':fn}


def load_checkpoint(filepath):
    
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    checkpoint = torch.load(filepath, map_location=map_location)
    
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model, checkpoint['class_names'], checkpoint['scan_plane']


def crop_background2D(image_np):

    ms = extract_middleSlice(image_np, 3)
    threshold = otsu(ms)
    output_data = image_np > threshold
    output_data = output_data.astype(int)
    img = image_np*output_data
    x,y,z = np.where(img>0)
    if image_np.shape[2] > 1:
        new_image = image_np[x.min():x.max(), y.min():y.max(),z.min():z.max()]
    else:
        new_image = image_np[x.min():x.max(), y.min():y.max(), :]
    return new_image

def load_reorient(fn):
    image = nib.load(fn)
    data = image.get_data()
    if len(data.shape)>3:
        data = data[:,:,:,0]
#    try:
#        data = data.astype('float32')
#    except:
#        print(fn)
#        raise Exception('RBG data!!!')
    data = crop_background2D(data)
    # take only volume 0
    sx, sy, sz = image.header.get_zooms()[:3]
    spacing = [sx,sy,sz]

    affine = image.affine
    data = np.expand_dims(data,axis = 0)
    array = data[np.newaxis]    
    if not nib.aff2axcodes(affine) == tuple('RAS'):
        # (1, C, W, H, D)
        # NIfTI images should have channels in 5th dimension
        array = array.transpose(2, 3, 4, 0, 1)
        # (W, H, D, 1, C)
        try:
            nii = nib.Nifti1Image(array, affine)
        except:
            print(fn)
        reoriented = nib.as_closest_canonical(nii)
        # https://nipy.org/nibabel/reference/nibabel.dataobj_images.html#nibabel.dataobj_images.DataobjImage.get_data
        array = np.asanyarray(reoriented.dataobj)
        # https://github.com/facebookresearch/InferSent/issues/99#issuecomment-446175325
        array = array.copy()
        array = array.transpose(3, 4, 0, 1, 2)  # (1, C, W, H, D)
        hd = reoriented.header
        spacing = hd.get_zooms()[:3]
        
    return image, array[0,:,:,:], spacing


class MRClassifierDataset_test(Dataset):

    def __init__(self, images, dummy, transform=None):
        """
        Args:
            
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.list_images = images
        self.dummy = dummy

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        
        img_name = self.list_images[idx]
        try:
            image = nib.load(img_name).get_data()
        except:
#             il = self.parent_dir+'/mr_class/random.nii.gz'
            image = nib.load(self.dummy).get_data()
            print('{0} seems to be corrupted'.format(img_name))
        

        if len(image.shape) > 3:
            #4D images, truncated to first volume
            image = image[:, :, :, 0]
        try:
            image = extract_middleSlice(image)
        except:
#             il = self.parent_dir +'/mr_class/random.nii.gz'
            image = nib.load(self.dummy).get_data()
            image = extract_middleSlice(image)
            print('{0} seems to be corrupted'.format(img_name))
        
        sample = {'image': image, 'name':img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample


class MRClassifierDataset_inference(Dataset):

    def __init__(self,list_images='', augmentations=None, 
                 class_names = '', run_3d = False, scan = 0, 
                 remove_corrupt = False, subclasses = False,
                 parentclass = False, spatial_size=224, 
                 nr_slices = 50, val = False, infer = False):
 
        self.list_images = list_images
        self.class_names = class_names
        self.augmentations = augmentations
        self.run_3d = run_3d
        self.scan = scan
        self.remove_corrupt = remove_corrupt
        self.subclasses = subclasses
        self.parentclass = parentclass
        self.nr_slices = nr_slices
        self.spatial_size = spatial_size
        self.val = val
        self.infer = infer
        
        
    def __len__(self):
        return len(self.list_images)
    
    def get_random(self,fa = False):
        
        if self.run_3d or fa:
            image = np.random.randn(2,self.spatial_size, self.spatial_size,self.nr_slices).astype('f')
        else:
            image = np.random.randn(self.spatial_size, self.spatial_size).astype('f')
        class_cat = 'random'
        return image, class_cat
    
    def __getitem__(self, idx):
        
        #modify the collate_fn from the dataloader so that it filters out None elements.
        img_name = self.list_images[idx]

        try:
            _, image, spacing = load_reorient(img_name)
        except:
            spacing = [2,2,2]
            print ('error loading {0}'.format(img_name))
            if self.remove_corrupt and os.path.isfile(img_name):
                os.remove(img_name)
            image, _ = self.get_random(fa=True)
        
        if not np.any(image):
            image, _ = self.get_random(fa=True)
        
        #preprocessing
        original_spacing = np.array(spacing)
        get_foreground = tio.ZNormalization.mean
        target_shape = 128, 128, 128
        crop_pad = tio.CropOrPad(target_shape)
        
        ###operations###
        standardize = tio.ZNormalization(masking_method=get_foreground)
        if 'wb' not in self.class_names and 'abd-pel' not in self.class_names:
            downsample = tio.Resample((2/spacing[0],2/spacing[1],2/spacing[2]))
            try:
                image = standardize(crop_pad(downsample(image))) 
            except:
                print(img_name)
        else:
            x,y,z = image.shape[1:]/np.asarray(target_shape)
            downsample = tio.Resample((x,y,z))
            image = standardize(crop_pad(downsample(image)))
            
        if image.shape[0] > 1:
            
            image = np.expand_dims(image[0, :, :, :], axis=0)
            #print(image.shape)
            print(img_name)

        sample = {'image': torch.from_numpy(image), 'label': np.array(0), 'spacing': original_spacing, 'fn': img_name}

        return sample
