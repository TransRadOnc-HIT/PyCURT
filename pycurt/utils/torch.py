import numpy as np
import torch
import cv2
from pycurt.utils.filemanip import extract_middleSlice
from torch.utils.data import Dataset
import nibabel as nib

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if sample is not None:
            image,name= sample['image'],sample['name']
            # torch.from_numpy(np.float32(image))
            return {'image': torch.from_numpy(np.float32(image)).unsqueeze(dim=0),
                    'name':name}

   
class ZscoreNormalization(object):
    """ put data in range of 0 to 1 """
   
    def __call__(self,sample):
        
        if sample is not None:
            np.seterr(divide='ignore', invalid='ignore')
            image,name= sample['image'], sample['name']
            image -= image.mean() 
            image /= image.std() 
            
        
                
            return {'image': image, 'name' : name}
        else:
            return None


class resize_2Dimage:
    """ Args: img_px_size slices resolution(cubic)
              slice_nr Nr of slices """
    
    def __init__(self,img_px_size):
        self.img_px_size=img_px_size
        
    def __call__(self,sample):
        image,name= sample['image'], sample['name']
        image_n= cv2.resize(image, (self.img_px_size, self.img_px_size),
                            interpolation=cv2.INTER_CUBIC)
        return {'image': image_n, 'name' : name}


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
    return model


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
