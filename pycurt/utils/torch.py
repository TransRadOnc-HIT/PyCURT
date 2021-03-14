import numpy as np
import torch
import cv2
from pycurt.utils.filemanip import extract_middleSlice
from torch.utils.data import Dataset
import nibabel as nib
import os

    
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
            image, label,spacing,fn = sample['image'], sample['label'], sample['spacing'], sample['fn']
            image -= image.mean() 
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


class MRClassifierDataset(Dataset):

    def __init__(self,list_images='', transform=None, augmentations=None, 
                 class_names = '', run_3d = False, scan = 0, 
                 remove_corrupt = True, subclasses = False,
                 parentclass = False, inference = False, spatial_size=224, 
                 nr_slices = 50):
 
        self.transform = transform
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
        
    def __len__(self):
        return len(self.list_images)
    
    def get_random(self):
        
        if self.run_3d:
            image = np.random.randn(self.spatial_size, self.spatial_size,self.nr_slices).astype('f')
        else:
            image = np.random.randn(self.spatial_size, self.spatial_size).astype('f')
        class_cat = 'random'
        return image, class_cat
    
    def __getitem__(self, idx):
        
        #modify the collate_fn from the dataloader so that it filters out None elements.
        img_name = self.list_images[idx]

        try:
            image = nib.load(img_name).get_data()
            if self.subclasses and self.parentclass:
                class_cat = img_name.split('/')[-2].split('_')[0]
            elif self.subclasses and not self.parentclass:
                class_cat = img_name.split('/')[-2].split('_')[-1]
            else:
                class_cat = img_name.split('/')[-2]
        except:
            print ('error loading {0}'.format(img_name))
            if self.remove_corrupt and os.path.isfile(img_name):
                os.remove(img_name)
            image, class_cat = self.get_random()
            
        if len(image.shape)>3:
            image = image[:,:,:,0]
            #print('4D images are not supported;{} is ignored'.format(img_name))
            
        if not self.run_3d:
            try:
                image = extract_middleSlice(image, self.scan)                 
            except:
                print ('error loading {0}'.format(img_name))
                if self.remove_corrupt and os.path.isfile(img_name):
                    os.remove(img_name)
                image, class_cat = self.get_random()
        spacing = image.shape
        
        label = 0
        if self.augmentations is not None:
            image = self.augmentations.augment_image(image)
        sample = {'image': image, 'label': np.array(label), 'spacing': spacing, 'fn': img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample
