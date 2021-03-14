import os
import pandas as pd
import math
import glob
import shutil
from pathlib import Path


ALLOWED_EXT = ['.xlsx', '.csv']
ILLEGAL_CHARACTERS = ['/', '(', ')', '[', ']', '{', '}', ' ', '-']


def mergedict(a, b):
    a.update(b)
    return a


def extract_middleSlice(image, scan):
    
    
    x,y,z = image.shape
    s = smallest(x,y,z)
    if (s==z and scan == 3) or scan == 2:
        ms= math.ceil(image.shape[2]/2)-1
        return image[:, :, ms].astype('float32')
    elif (s==y and scan == 3) or scan == 1:
        ms=math.ceil(image.shape[1]/2)-1
        return image[:, ms, :].astype('float32')
    else:
        ms= math.ceil(image.shape[0]/2)-1
        return image[ms, :, :].astype('float32')
    

def extract_middleSlice_old(image):
    
    x, y, z = image.shape
    s = smallest(x,y,z)
    if s == z:
        ms = math.ceil(image.shape[2]/2)-1
        return image[:, :, ms].astype('float32')
    elif s == y:
        ms = math.ceil(image.shape[1]/2)-1
        return image[:, ms, :].astype('float32')
    else:
        ms = math.ceil(image.shape[0]/2)-1
        return image[ms, :, :].astype('float32')


def smallest(num1, num2, num3):
    
    if (num1 < num2) and (num1 < num3):
        smallest_num = num1
    elif (num2 < num1) and (num2 < num3):
        smallest_num = num2
    else:
        smallest_num = num3
    return smallest_num


def label_move_image(image, modality, out_dir, renaming=True):

    sub_name, tp = image.split('/')[-3:-1]
    base_dir_path = os.path.join(out_dir, sub_name, tp)
    dir_name = os.path.join(base_dir_path, modality)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    if renaming:
        new_name = file_rename(image)
    else:
        new_name = image
    try:
        shutil.copytree(new_name, os.path.join(dir_name, new_name.split('/')[-1]))
        outname = os.path.join(dir_name, new_name.split('/')[-1])
    except:
        files = [item for item in glob.glob(dir_name+'/*')
                 if new_name.split('/')[-1] in item ]
        if len(files) == 1:
            new_name1 = new_name+'_1'
        else:
            new_name1 = new_name+'_'+ str(int(len(files)))
        # Renaming old directory
        shutil.move(new_name, new_name1)  
        # Copy to the sorting location  
        shutil.copytree(new_name1, os.path.join(dir_name, new_name1.split('/')[-1]))
        outname = os.path.join(dir_name, new_name1.split('/')[-1])
        new_name = new_name1
    
    return outname, new_name

def file_rename(image):

    base_dir_path = os.path.split(image)[0]
    name = image.split('/')[-1]
    name_parts = name.split('-')
    if not name_parts[0]:
        new_name = os.path.join(base_dir_path, 'image')
        shutil.move(image, new_name)
    elif len(name_parts) > 1:
        if os.path.isdir(os.path.join(base_dir_path, name_parts[0])):
            new_name = os.path.join(base_dir_path, name_parts[0][:-1])
        else:
            new_name = os.path.join(base_dir_path, name_parts[0])
        shutil.move(image, new_name)
    else:
        new_name = image
    return new_name


def create_move_toDir(fileName, dirName, actRange):
    
    folderName=Path(fileName[0:-7])
    indices = [i for i, x in enumerate(folderName.parts[-1]) if x == "-"]
    indices2=[i for i, x in enumerate(dirName) if x == "/"]
    
    if not os.path.exists(dirName) and not os.path.isdir(dirName):
        os.makedirs(dirName)
        print(folderName)
        print(indices)
        try:
            newName = os.path.join(dirName[0:indices2[-1]],
                                   '1-'+ folderName.parts[-1][0:indices[0]]+'-'+str(actRange))
        except IndexError:
            newName = os.path.join(dirName[0:indices2[-1]],
                                   '1-'+ folderName.parts[-1]+'-'+str(actRange))
       
    else:
        f = [item for item in os.listdir(dirName) if '1-' in item]
        if len(f) > 1:
            [f.remove(x) for x in f if len(x.split('-')) != 3]
        if len(f) > 1:
            for ff in f[1:]:
                if os.path.isfile(ff):
                    os.remove(ff)
        indices3 = [i for i, x in enumerate(f[0]) if x == "-"]
        actRange_f = float(f[0][indices3[-1]+1:])
#         actRange_f = sorted([float(x[indices3[-1]+1:]) for x in f])[0]

        if actRange>actRange_f:
            try:
                newName=os.path.join(dirName[0:indices2[-1]],
                                     '1-'+ folderName.parts[-1][0:indices[0]]+'-'+str(actRange))
            except:
                newName=os.path.join(dirName[0:indices2[-1]],
                                     '1-'+ folderName.parts[-1]+'-'+str(actRange))
            if not os.path.isdir(os.path.join(dirName,f[0][indices3[0]+1:])):
                shutil.move(os.path.join(dirName,f[0]),
                            os.path.join(dirName,f[0][indices3[0]+1:]))
            else:
                shutil.move(os.path.join(dirName,f[0]),
                            os.path.join(dirName,f[0][indices3[0]+1:]+'_1'))
          
        elif actRange <= actRange_f:
            try:
                newName=os.path.join(dirName[0:indices2[-1]],
                                     folderName.parts[-1][0:indices[0]]+'-'+str(actRange))
            except IndexError:
                newName=os.path.join(dirName[0:indices2[-1]],
                                     folderName.parts[-1]+'-'+str(actRange))
#     shutil.move(fileName[0:-7],newName)
    shutil.copytree(fileName[0:-7], newName)
    try:
        shutil.move(newName, dirName)
    except:
        files=[item for item in glob.glob(dirName+'/*') if newName.split('/')[-1] in item ]
        if len(files) == 1:
            newName1=newName+'_1'
        else:
            newName1=newName+'_'+ str(int(len(files)))
            
        shutil.move(newName, newName1)    
        shutil.move(newName1, dirName)


def strip_non_ascii(string):
        ''' Returns the string without non ASCII characters'''
        stripped = (c for c in string if 0 < ord(c) < 127)
        return ''.join(stripped)
