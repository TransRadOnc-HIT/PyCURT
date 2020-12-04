import pydicom
import numpy as np
from operator import itemgetter
import collections
from pydicom.errors import InvalidDicomError
import os
import subprocess as sp
from pathlib import Path


ExplicitVRLittleEndian = '1.2.840.10008.1.2.1'
ImplicitVRLittleEndian = '1.2.840.10008.1.2'
DeflatedExplicitVRLittleEndian = '1.2.840.10008.1.2.1.99'
ExplicitVRBigEndian = '1.2.840.10008.1.2.2'
NotCompressedPixelTransferSyntaxes = [
    ExplicitVRLittleEndian, ImplicitVRLittleEndian,
    DeflatedExplicitVRLittleEndian, ExplicitVRBigEndian]


class DicomInfo(object):
    
    def __init__(self, dicoms):

        if type(dicoms) == list:
            self.dcms = dicoms
        elif os.path.isdir(dicoms):
            dcms = list(dicoms.glob('*.dcm'))
            if not dcms:
                dcms = list(dicoms.glob('*.IMA'))
            if not dcms:
                raise Exception('No DICOM files found in {}'.format(dicoms))
            else:
                self.dcms = dcms
        else:
            self.dcms = [dicoms]

    def get_tag(self, tag):
        
        tags = {}

        if type(tag) is not list:
            tag = [tag]
        for t in tag:
            values = []
            for dcm in self.dcms:
                header = pydicom.read_file(str(dcm))
                try:
                    val = header.data_element(t).value
                    if isinstance(val, collections.Iterable) and type(val) is not str:
                        val = tuple(val)
                    else:
                        val = str(val)
                    values.append(val)
                except (AttributeError, KeyError):
                    print ('{} seems to do not have the requested DICOM field ({})'.format(dcm, t))

            tags[t] = list(set(values))
        
        return self.dcms, tags

    def check_uniqueness(self, InstanceNums, SeriesNums):
        
        toRemove = []
        if (len(InstanceNums) == 2*(len(set(InstanceNums)))) and len(set(SeriesNums)) == 1:
            sortedInstanceNums = sorted(zip(self.dcms, InstanceNums), key=itemgetter(1))
            uniqueInstanceNums = [x[0] for x in sortedInstanceNums[:][0:-1:2]]
            toRemove = toRemove+uniqueInstanceNums
        
        return toRemove


def dcm_info(dcm_folder):
    """Function to extract information from a list of DICOM files in one folder. It returns a list of
    unique image types and scan numbers found in the input list of DICOMS.
    Parameters
    ----------
    dcm_folder : str
        path to an existing folder with DICOM files
    Returns
    -------
    dicoms : list
        list of DICOM files in the folder
    image_types : list
        list of unique image types extracted from the DICOMS
    series_nums : list
        list of unique series numbers extracted from the DICOMS
    """
    dcm_folder = Path(dcm_folder)

    dicoms = sorted(list(dcm_folder.glob('*.dcm')))
    if not dicoms:
        dicoms = sorted(list(dcm_folder.glob('*.IMA')))
        if not dicoms:
            raise Exception('No DICOM files found in {}'.format(dcm_folder))

    ImageTypes = []
    SeriesNums = []
    toRemove = []
    InstanceNums = []
    for dcm in dicoms:
        #Check whether the dicom is compressed, if yes decompress
        try:
            if (pydicom.read_file(str(dcm)).file_meta.TransferSyntaxUID
                    not in NotCompressedPixelTransferSyntaxes):
                decompress_dicom(dcm)
#         except InvalidDicomError:
#             print ('{} seems to do not have a readable DICOM header and '
#                    'will be removed from the folder'.format(dcm))
#             toRemove.append(dcm)
#         try:
            header = pydicom.read_file(str(dcm))
            ImageTypes.append(tuple(header.ImageType))
            SeriesNums.append(header.SeriesNumber)
            InstanceNums.append(header.InstanceNumber)
        except AttributeError:
            print ('{} seems to do not have the right DICOM fields and '
                   'will be removed from the folder'.format(dcm))
            toRemove.append(dcm)
        except InvalidDicomError:
            print ('{} seems to do not have a readable DICOM header and '
                   'will be removed from the folder'.format(dcm))
            toRemove.append(dcm)
    # the following lines are to check whether or not there are 2 set of exactly the same DICOM files in the folder
    if (len(InstanceNums) == 2*(len(set(InstanceNums)))) and len(set(SeriesNums)) == 1:
        sortedInstanceNums = sorted(zip(dicoms, InstanceNums), key=itemgetter(1))
        uniqueInstanceNums = [x[0] for x in sortedInstanceNums[:][0:-1:2]]
        toRemove = toRemove+uniqueInstanceNums
    if toRemove:
        for f in toRemove:
            dicoms.remove(f)

    return dicoms, list(set(ImageTypes)), list(set(SeriesNums))


def dcm_check(dicoms, im_types, series_nums):
    """Function to check the DICOM files in one folder. It is based on the glioma test data.
    This function checks the type of the image (to exclude those that are localizer acquisitions)
    and the series number (if in one folder there are more than one scans then this function will
    return the second one, assuming that it is the one after the contrast agent injection).
    It returns a list of DICOMS which belong to one scan only, ignoring localizer scans. 
    Parameters
    ----------
    dicoms : list
        list of DICOMS in one folder
    im_types : list
        list of all image types extracted from the DICOM headers
    series_nums : list
        list of all scan numbers extracted from the DICOM headers
    Returns
    -------
    dcms : list
        list of DICOMS files
    """
    if len(im_types) > 1:
        try:
            im_type = list([x for x in im_types if not 'PROJECTION IMAGE' in x
                            and 'LOCALIZER' not in x][0])
    
            dcms = [x for x in dicoms if pydicom.read_file(str(x)).ImageType==im_type]
        except IndexError:
            dcms = []
    elif len(series_nums) > 1:
        series_num = np.max(series_nums)
        dcms = [x for x in dicoms if pydicom.read_file(str(x)).SeriesNumber==series_num]
    else:
        dcms = dicoms

    return [str(x) for x in dcms]


def decompress_dicom(dicom):
    
    cmd = ("gdcmconv --raw {0} {1} ".format(dicom, dicom))
    sp.check_output(cmd, shell=True) 
