from nipype.interfaces.base import (
    BaseInterface, TraitedSpec, Directory, File,
    traits, BaseInterfaceInputSpec, InputMultiPath)
import pydicom
import numpy as np
from pycurt.utils.dicom import dcm_check, dcm_info
from pathlib import Path
import shutil
import os
import nibabel as nib
import glob
from core.utils.filemanip import split_filename
from pycurt.utils.filemanip import label_move_image
from skimage.transform import resize
import pydicom as pd
import re
from collections import defaultdict
from nipype.interfaces.base import isdefined
from pycurt.converters.dicom import DicomConverter
from . import logging
import SimpleITK as sitk
from datetime import datetime as dt
from datetime import timedelta
from pycurt.utils.filemanip import create_move_toDir


iflogger = logging.getLogger('nipype.interface')


ILLEGAL_CHARACTERS = ['/', '(', ')', '[', ']', '{', '}', ' ']
RT_NAMES = ['RTSTRUCT', 'RTDOSE', 'RTPLAN', 'RTCT']
POSSIBLE_NAMES = ['RTSTRUCT', 'RTDOSE', 'RTPLAN', 'T1KM', 'FLAIR',
                  'CT', 'ADC', 'T1', 'SWI', 'T2', 'T2KM', 'CT1',
                  'RTCT']
ExplicitVRLittleEndian = '1.2.840.10008.1.2.1'
ImplicitVRLittleEndian = '1.2.840.10008.1.2'
DeflatedExplicitVRLittleEndian = '1.2.840.10008.1.2.1.99'
ExplicitVRBigEndian = '1.2.840.10008.1.2.2'
NotCompressedPixelTransferSyntaxes = [ExplicitVRLittleEndian,
                                      ImplicitVRLittleEndian,
                                      DeflatedExplicitVRLittleEndian,
                                      ExplicitVRBigEndian]

class DicomCheckInputSpec(BaseInterfaceInputSpec):

    dicom_dir = Directory(exists=True, desc='Directory with the DICOM files to check')
    working_dir = Directory('checked_dicoms', usedefault=True,
                            desc='Base directory to save the corrected DICOM files')


class DicomCheckOutputSpec(TraitedSpec):

    outdir = Directory(exists=True, desc='Path to the directory with the corrected DICOM files')
    scan_name = traits.Str(desc='Scan name')
    dose_file = File(desc='Dose file, if any')
    dose_output = File()


class DicomCheck(BaseInterface):

    input_spec = DicomCheckInputSpec
    output_spec = DicomCheckOutputSpec

    def _run_interface(self, runtime):

        dicom_dir = self.inputs.dicom_dir
        wd = os.path.abspath(self.inputs.working_dir)
        self.dose_file = None

        img_paths = dicom_dir.split('/')
        scan_name = list(set(POSSIBLE_NAMES).intersection(img_paths))[0]
        name_index = img_paths.index(scan_name)
        tp = img_paths[name_index-1]
        sub_name = img_paths[name_index-2]
        if scan_name in RT_NAMES:
            if scan_name == 'RTDOSE':
                scan_name = scan_name+'_{}.nii.gz'.format(img_paths[-1])
            dicoms = sorted(os.listdir(dicom_dir))
            if not os.path.isdir(wd):
                os.makedirs(wd)
            for item in dicoms:
                curr_item = os.path.join(dicom_dir, item)
                if os.path.isdir(curr_item):
                    shutil.copytree(curr_item, wd)
                else:
                    shutil.copy2(curr_item, os.path.join(wd, item))
                if scan_name == 'RTSTRUCT':
                    rt_dcm = glob.glob(wd+'/*.dcm')[0]
                    ds = pd.read_file(rt_dcm)
                    regex = re.compile('[^a-zA-Z]')
                    for i in range(len(ds.StructureSetROISequence)):
                        new_roiname=regex.sub('', ds.StructureSetROISequence[i].ROIName)
                        ds.StructureSetROISequence[i].ROIName = new_roiname
                    ds.save_as(rt_dcm)
        else:
            dicoms, im_types, series_nums = dcm_info(dicom_dir)
            dicoms = dcm_check(dicoms, im_types, series_nums)
            if dicoms:
                if not os.path.isdir(wd):
                    os.makedirs(wd)
                    for d in dicoms:
                        shutil.copy2(d, wd)
        self.outdir = wd
        self.scan_name = scan_name
        if 'RTDOSE' in scan_name:
            self.dose_file = glob.glob(os.path.join('/'.join(img_paths), '*.dcm'))[0]
            self.dose_output = os.path.join(wd, sub_name, tp, '{}.nii.gz'.format(scan_name))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['outdir'] = self.outdir
        outputs['scan_name'] = self.scan_name
        if self.dose_file is not None:
            outputs['dose_file'] = self.dose_file
            outputs['dose_output'] = self.dose_output
        else:
            outputs['dose_file'] = self.scan_name
            outputs['dose_output'] = self.scan_name

        return outputs


class ConversionCheckInputSpec(BaseInterfaceInputSpec):

    in_file = InputMultiPath(File(), desc='(List of) file that'
                             ' needs to be checked after DICOM to NIFTI conversion')
    file_name = traits.Str(desc='Name that the converted file has to match'
                           ' in order to be considered correct.')


class ConversionCheckOutputSpec(TraitedSpec):

    out_file = traits.Str()


class ConversionCheck(BaseInterface):

    input_spec = ConversionCheckInputSpec
    output_spec = ConversionCheckOutputSpec

    def _run_interface(self, runtime):

        converted = self.inputs.in_file
        scan_name = self.inputs.file_name
        
        converted_old = converted[:]
        to_remove = []
        base_dir = os.path.dirname(converted[0])
        extra = [x for x in converted if x.split('/')[-1]!='{}.nii.gz'.format(scan_name)]
        if len(extra) == len(converted):
            if len(extra) == 2 and scan_name == 'T2':
                to_remove += extra
                if not os.path.isfile(os.path.join(base_dir, 'T2.nii.gz')):
                    shutil.copy2(extra[1], os.path.join(base_dir, 'T2.nii.gz'))
                converted_old.append(os.path.join(base_dir, 'T2.nii.gz'))
            else:
                to_remove += extra
                converted = None
        else:
            to_remove += extra

        if to_remove:
            for f in to_remove:
                if os.path.isfile(f):
                    converted_old.remove(f)
        if converted_old:
            self.converted = converted_old[0]
            try:
                ref = nib.load(self.converted)
                data = ref.get_data()
                if len(data.squeeze().shape) == 2 or len(data.squeeze().shape) > 4:
                    if os.path.isfile(self.converted):
                        self.converted = None
                elif len(data.squeeze().shape) == 4:
                    im2save = nib.Nifti1Image(data[:, :, :, 0], affine=ref.affine)
                    nib.save(im2save, self.converted)
                elif len(data.dtype) > 0:
                    iflogger.info('{} is not a greyscale image. It will'
                                  ' be deleted.'.format(self.converted))
                    if os.path.isfile(self.converted):
                        self.converted = None
            except:
                iflogger.info('{} failed to save with nibabel. '
                              'It will be deleted.'.format(self.converted))
                if os.path.isfile(self.converted):
                    self.converted = None
        else:
            self.converted = None

        if self.inputs.in_file and self.converted is None:
            outfile = self.inputs.in_file[0].split('.nii')[0]+'_WRONG_CONVERTION.nii.gz'
            shutil.copy2(self.inputs.in_file[0], outfile)
            self.converted = outfile

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        if self.converted is not None:
            outputs['out_file'] = self.converted

        return outputs


class RemoveRTFilesInputSpec(BaseInterfaceInputSpec):

    source_dir = traits.List()
    out_filename = traits.List()
    output_dir = traits.List()


class RemoveRTFilesOutputSpec(TraitedSpec):

    source_dir = traits.List()
    out_filename = traits.List()
    output_dir = traits.List()


class RemoveRTFiles(BaseInterface):
    
    input_spec = RemoveRTFilesInputSpec
    output_spec = RemoveRTFilesOutputSpec
    
    def _run_interface(self, runtime):
        
        source_dir = self.inputs.source_dir
        out_filename = self.inputs.out_filename
        output_dir = self.inputs.output_dir
        
        indexes = [i for i, x in enumerate(out_filename) if 'RTSTRUCT' not in x]
        self.source_dir = [source_dir[x] for x in indexes]
        self.out_filename = [out_filename[x] for x in indexes]
        self.output_dir = [output_dir[x] for x in indexes]
        
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['source_dir'] = self.source_dir
        outputs['out_filename'] = self.out_filename
        outputs['output_dir'] = self.output_dir

        return outputs


class NNUnetPreparationInputSpec(BaseInterfaceInputSpec):

    images = traits.List(mandatory=True, desc='List of images to be prepared before'
                         ' running the nnUNet inference.')


class NNUnetPreparationOutputSpec(TraitedSpec):

    output_folder = Directory(exists=True, desc='Output folder prepared for nnUNet.')


class NNUnetPreparation(BaseInterface):

    input_spec = NNUnetPreparationInputSpec
    output_spec = NNUnetPreparationOutputSpec

    def _run_interface(self, runtime):

        images = self.inputs.images
        if images:
            new_dir = os.path.abspath('data_prepared')
            os.mkdir(os.path.abspath('data_prepared'))
            for i, image in enumerate(images):
                _, _, ext = split_filename(image)
                shutil.copy2(image, os.path.join(
                    new_dir,'subject1_{}'.format(str(i).zfill(4))+ext))
        else:
            raise Exception('No images provided!Please check.')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_folder'] = os.path.abspath('data_prepared')

        return outputs


class CheckRTStructuresInputSpec(BaseInterfaceInputSpec):
    
    rois = InputMultiPath(File(exists=True), desc='RT structures to check')
    dose_file = File(exists=True, desc='Dose file.')


class CheckRTStructuresOutputSpec(TraitedSpec):
    
    checked_roi = File(exists=True, desc='ROI with the maximum overlap with the dose file.')


class CheckRTStructures(BaseInterface):
    
    input_spec = CheckRTStructuresInputSpec
    output_spec = CheckRTStructuresOutputSpec

    def _run_interface(self, runtime):
    
        rois = self.inputs.rois
        dose_nii = self.inputs.dose_file
        rois1 = [x for x in rois if 'gtv' in x.split('/')[-1].lower()]
        if not rois1:
            rois1 = [x for x in rois if 'ptv' in x.split('/')[-1].lower()]
        if not rois1:
            rois1 = [x for x in rois if 'ctv' in x.split('/')[-1].lower()]
        if not rois:
            raise Exception('No GTV, PTV or CTV found in the rois! Please check')

        if len(rois1) > 1:
            roi_dict = {}
            dose = nib.load(dose_nii).get_data()
            dose_vector = dose[dose > 0]
            dose_maxvalue = np.percentile(dose_vector, 99)
            ref_roi = nib.load(rois1[0]).get_data()
            if dose.shape != ref_roi.shape:
                dose = resize(dose, ref_roi.shape, order=0, mode='edge',
                                   cval=0, anti_aliasing=False)

            dose_bool = dose >= dose_maxvalue
            
            for f in rois1:
                roi = nib.load(f).get_data()
                roi_bool = roi > 0
                nr_andvoxel = np.logical_and(dose_bool, roi_bool)
                roi_dict[f] = np.sum(nr_andvoxel)/np.sum(roi_bool)
            roi_tokeep = max(roi_dict, key=lambda key: roi_dict[key])
            for key in roi_dict.keys():
                if key == roi_tokeep:
                    self.checked_roi = key
        elif len(rois1) == 1:
            self.checked_roi = rois1[0]

        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['checked_roi'] = self.checked_roi

        return outputs


class GetRefRTDoseInputSpec(BaseInterfaceInputSpec):
    
    doses = InputMultiPath(Directory(exists=True), desc='RT doses to check')


class GetRefRTDoseOutputSpec(TraitedSpec):
    
    dose_file = File(exists=True, desc='Dose file to be converted.')


class GetRefRTDose(BaseInterface):
    
    input_spec = GetRefRTDoseInputSpec
    output_spec = GetRefRTDoseOutputSpec

    def _run_interface(self, runtime):
    
        doses = self.inputs.doses
        phys = [x for y in doses for x in glob.glob(y+'/*/*.dcm') if 'PHY' in x]
        rbe = [x for y in doses for x in glob.glob(y+'/*/*.dcm') if 'RBE' in x]
        if phys:
#             dcms = glob.glob(phys[0]+'/*.dcm')
            dcms = phys
        elif rbe:
            dcms = rbe
        elif doses: 
            dcms = [x for y in doses for x in glob.glob(y+'/*.dcm')]

        right_dcm = []
        for dcm in dcms:
            hd = pydicom.read_file(dcm)
            try:
                hd.GridFrameOffsetVector
                right_dcm.append(dcm)
            except:
                continue

        dcms = right_dcm[:]
        if dcms and len(dcms)==1: 
            dose_file = dcms[0]
        elif dcms and len(dcms) > 1: 
            iflogger.info('More than one dose file') 
            processed = False 
            for dcm in dcms: 
                hd = pydicom.read_file(dcm) 
                dose_tp = hd.DoseSummationType 
                if not 'BEAM' in dose_tp and not processed: 
                    dose_file = dcm
                    processed = True 
                    break 
            if not processed:
                iflogger.info('No PLAN in any dose file')
                dose_file = dcms[0]
        self.dose_file = dose_file
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['dose_file'] = self.dose_file

        return outputs


class FileCheckInputSpec(BaseInterfaceInputSpec):
    
    input_dir = Directory(exists=True, desc='Input directory to prepare properly.')
    renaming = traits.Bool(False, desc='Whether or not to use the information stored'
                           'in the DICOM header to rename the subject and sessions '
                           'folders. If False, the file path will be splitted '
                           'and the subject name will be taken from there. In this '
                           'case, the subject_name_position must be provided.'
                           'Default is False.', usedefault=True)
    subject_name_position = traits.Int(
        -3, usedefault=True, desc='The position of the subject name in the splitted '
        'file path (file_path.split("/")). Default is -3, so it assumes that the subject '
        'name is in the third position starting from the end of the file path.')


class FileCheckOutputSpec(TraitedSpec):
    
    out_list = traits.List(desc='Prepared folder.')


class FileCheck(BaseInterface):
    
    input_spec = FileCheckInputSpec
    output_spec = FileCheckOutputSpec
    
    def _run_interface(self, runtime):

        input_dir = self.inputs.input_dir
        renaming = self.inputs.renaming
        out_list = []
        if not renaming:
            sub_name_position = self.inputs.subject_name_position

        scans = defaultdict(list)
        patient_names = defaultdict(list)
        scan_dates = defaultdict(list)
        z = 0
        for path, _, files in os.walk(input_dir):
            for f in files:
                if '.dcm' in f or '.ima' in f.lower():
                    filename = os.path.join(path, f)
#                     iflogger.info('Process number: {}\n File: {}'.format(z, filename))
                    try:
                        ds = pydicom.dcmread(filename, force = True)
                    except:
                        iflogger.info('{} could not be read, dicom '
                              'file may be corrupted'.format(filename))
                    try:
                        seriesDescription=ds.SeriesDescription.upper().replace('_','')
                    except:
                        try:
                            seriesDescription=ds.Modality.upper().replace('_','')
                        except:
                            seriesDescription='NONE'
                    try:
                        studyInstance = ds.StudyInstanceUID
                    except:
                        studyInstance='NONE'
                    try:
                        seriesInstance = ds.SeriesInstanceUID
                    except:
                        seriesInstance='NONE'
                    key = seriesDescription +'_' + seriesInstance + '_' + studyInstance
                    key = self.strip_non_ascii(re.sub(r'[^\w]', '', key))
                    key = key.replace('_','-')
                    scans[key].append(filename)
                    if renaming:
                        try:
                            patient_names[key].append(ds.PatientID)
                        except AttributeError:
                            iflogger.info('No patient ID for {}'.format(filename))
                            patient_names[key].append('Corrupted')
                    else:
                        sub_name = filename.split('/')[sub_name_position]
                        patient_names[key].append(sub_name)
                    try:
                        scan_dates[key].append(ds.StudyDate)
                    except:
                        iflogger.info('No study date for {}'.format(filename))
                        scan_dates[key].append('Corrupted')
                    z += 1
        names = [patient_names[x][0] for x in patient_names.keys()]
        for s in set(names):
            temp_scan = {}
            temp_pn = {}
            temp_sd = {}
            for key in patient_names.keys():
                if patient_names[key][0] == s:
                    temp_scan[key] = scans[key]
                    temp_pn[key] = patient_names[key]
                    temp_sd[key] = scan_dates[key]
            out_list.append([temp_scan, temp_pn, temp_sd])

        self.out_list = out_list

        return runtime

    def strip_non_ascii(self, string):
        ''' Returns the string without non ASCII characters'''
        stripped = (c for c in string if 0 < ord(c) < 127)
        return ''.join(stripped)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_list'] = self.out_list

        return outputs




class FolderPreparationInputSpec(BaseInterfaceInputSpec):
    
    input_list = traits.List(desc='Input directory to prepare properly.')
    out_folder = Directory('prepared_dir', usedefault=True,
                           desc='Prepared folder.')


class FolderPreparationOutputSpec(TraitedSpec):

    out_folder = Directory(exists=True, desc='Prepared folder.')
    for_inference_ct = traits.Dict(help='Dictionary of images to be classified using '
                                'mrclass or bpclass.')
    for_inference_mr = traits.Dict(help='Dictionary of images to be classified using '
                                'mrclass or bpclass.')
    to_crop = traits.List(help='CT image to crop in case of preclinical study.')


class FolderPreparation(BaseInterface):
    
    input_spec = FolderPreparationInputSpec
    output_spec = FolderPreparationOutputSpec
    
    def _run_interface(self, runtime):
        
        input_list = self.inputs.input_list
        output_dir = os.path.abspath(self.inputs.out_folder)
        
        scans = input_list[0]
        patient_names = input_list[1]
        scan_dates = input_list[2]
        for key in scans.keys():
            for file in scans[key]:
                pn = patient_names[key][0]
                for character in ILLEGAL_CHARACTERS:
                    pn = pn.replace(character, '_')
                out_basename = os.path.join(pn, scan_dates[key][0])
                dir_name = os.path.join(output_dir, out_basename, key)
                if not os.path.isdir(dir_name):
                    os.makedirs(dir_name)
                shutil.copy2(Path(file), dir_name)

        self.for_inference_ct, self.for_inference_mr = self.sort(output_dir)

        return runtime

    def sort(self, input_dir):

        out_dir = input_dir
        modality_list_rt = ['RTDOSE', 'RTSTRUCT', 'RTPLAN', 'PET', 'CT']
        modality_list_inference = ['MR', 'OT']
        
        images = glob.glob(input_dir+'/*/*/*')
        for_inference_mr = {}
        for_inference_ct = {}
        for_inference_mr['MR'] = []
        for_inference_ct['CT'] = []

        for i in images:
            if os.path.isdir(i):
                dcm_files = [os.path.join(i, item) for item in os.listdir(i)
                             if ('.dcm' in item or '.ima' in item.lower())]
            else:
                continue
            try:
                ds = pydicom.dcmread(dcm_files[0], force=True)
                modality_check = ds.Modality
            except:
                modality_check = ''
            if modality_check == 'RTSS':
                modality_check = 'RTSTRUCT'
            if modality_check in modality_list_rt:
                new_image, i = label_move_image(i, modality_check, out_dir,
                                                renaming=False)
                if modality_check == 'CT':
                    if len(dcm_files) > 20:
                        converter = DicomConverter(new_image)
                        nifti_image = converter.convert(rename_dicom=True, force=True)
                        if nifti_image is not None:
                            for_inference_ct['CT'].append(nifti_image)
#                     else:
#                         label_move_image(i, 'error_converting', out_dir,
#                                          renaming=False)
#                         iflogger.info('Error converting', str(new_image))
            elif modality_check in modality_list_inference:
                #checking for duplicates or localizer
                if len(dcm_files) > 9:
                    new_image, i = label_move_image(i, '', out_dir,
                                                    renaming=False)
                    dicoms, im_types, series_nums = dcm_info(new_image)
                    dicoms_fl = dcm_check(dicoms, im_types, series_nums)
    #                 corrupted = [f for f in dicoms if str(f) not in dicoms_fl]
                    [os.remove(f) for f in dicoms if str(f) not in dicoms_fl]
                    good = [f for f in dicoms if str(f) in dicoms_fl]
    #                 [os.remove(f) for f in dicoms if str(f) not in dicoms_fl]
                    if good:
                        converter = DicomConverter(new_image)
                        nifti_image = converter.convert(rename_dicom=True)
                        if nifti_image is not None:
                            for_inference_mr['MR'].append(nifti_image)
#                     else:
#                         label_move_image(i, 'error_converting', out_dir,
#                                          renaming=False)
#                         iflogger.info('Error converting', str(new_image))
            else:
                label_move_image(i, 'Unknown_modality', out_dir, renaming=False)

        return for_inference_ct, for_inference_mr

    def strip_non_ascii(self, string):
        ''' Returns the string without non ASCII characters'''
        stripped = (c for c in string if 0 < ord(c) < 127)
        return ''.join(stripped)

    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.out_folder):
            outputs['out_folder'] = os.path.abspath(
                self.inputs.out_folder)
            outputs['for_inference_ct'] = self.for_inference_ct
            outputs['for_inference_mr'] = self.for_inference_mr
            outputs['to_crop'] = self.for_inference_ct['CT']

        return outputs


class FolderMergeInputSpec(BaseInterfaceInputSpec):
    
    input_list = traits.List(help='Input directory to sort.')
    ct_dict = traits.Dict(help='Dictionary with labelled CTs to sink')
    rt_dict = traits.Dict(help='Dictionary with labelled RTs to sink')
    mr_dict = traits.Dict(help='Dictionary with labelled MRIs to sink')
    rert_max_time = traits.Int(help='Any RT session within this time (in days)'
                               'from the first RT will be considered as'
                               'replanning RT.')
    mrrt_max_time_diff = traits.Int(help='Max time, in days, after the RT session '
                                    'in which PyCURT is expecting to find an MR '
                                    'planning session.')
    out_folder = Directory('Sorted_Data', usedefault=True,
                           desc='Prepared folder.')


class FolderMergeOutputSpec(TraitedSpec):
    
    out_folder = Directory(help='Sorted folder.')


class FolderMerge(BaseInterface):
    
    input_spec = FolderMergeInputSpec
    output_spec = FolderMergeOutputSpec
    
    def _run_interface(self, runtime):

        input_list = self.inputs.input_list
        rert_max_time = self.inputs.rert_max_time
        mrrt_max_time_diff = self.inputs.mrrt_max_time_diff
        out_dir = os.path.abspath(self.inputs.out_folder)
        toreprocess = []

        for directories in input_list:
            session_dict = {}
            session_dict['RT'] = []
            session_dict['CT'] = []
            session_dict['MR'] = []
            rt_dict = directories[1]
            ct_dict = directories[0]
            if len(directories) == 3:
                mr_dict = directories[2]
            else:
                mr_dict = None
            input2sort = [[x, y] for x, y in zip(['CT', 'MR'], [ct_dict, mr_dict])
                          if (y is not None and y)]
            sub_info = {}
            for modality, outdict in input2sort:
                sub_info[modality] = {}
                sub_info[modality]['sub_name'] = []
                sub_info[modality]['sessions'] = []
                for key in outdict[modality].keys():
                    for cm in outdict[modality][key]:
                        indices = [i for i, x in enumerate(cm[0]) if x == "/"]
                        if modality == 'MR':
                            sub_info[modality]['sub_name'].append(cm[0].split('/')[-3])
                            sub_info[modality]['sessions'].append(cm[0].split('/')[-2])
                            dirName = os.path.join(
                                out_dir, cm[0][indices[-3]+1:indices[-1]], key)
                        else:
                            sub_info[modality]['sub_name'].append(cm[0].split('/')[-4])
                            sub_info[modality]['sessions'].append(cm[0].split('/')[-3])
                            dirName = os.path.join(
                                out_dir, cm[0][indices[-4]+1:indices[-1]])
                        create_move_toDir(cm[0], dirName, cm[1])
                sub_info[modality]['sub_name'] = list(set(sub_info[modality]['sub_name']))
                sub_info[modality]['sessions'] = list(set(sub_info[modality]['sessions']))
                if len(sub_info[modality]['sub_name']) > 1:
                    raise Exception('Multiple subject names found. Something went wrong.')
                for folder_name in sub_info[modality]['sessions']:
                    sub_name = sub_info[modality]['sub_name'][0]
                    session_dict[modality].append([os.path.join(
                        out_dir, sub_name, folder_name),
                        dt.strptime(folder_name, '%Y%m%d')])
            rt_sub_names = []
            for rt_tp in rt_dict.keys():
                sub_name, session = rt_tp.split('_')
                rt_sub_names.append(sub_name)
                session_dict['RT'].append([os.path.join(
                    out_dir, sub_name, session+'_RT'),
                    dt.strptime(session, '%Y%m%d')])
                session = session+'_RT'
                if 'rtplan' in rt_dict[rt_tp].keys():
                    rtplan_dir = os.path.join(
                        out_dir, sub_name, session, 'RTPLAN')
                    os.makedirs(os.path.join(rtplan_dir, '1-RTPLAN_Used'),
                                exist_ok=True)
                    shutil.copy2(rt_dict[rt_tp]['rtplan'],
                                 os.path.join(rtplan_dir, '1-RTPLAN_Used'))
                    if rt_dict[rt_tp]['other_rtplan']:
                        other_dir = os.path.join(rtplan_dir, 'Other_RTPLAN')
                        os.makedirs(other_dir)
                        [shutil.copytree(x, os.path.join(other_dir, x.split('/')[-1]))
                         for x in rt_dict[rt_tp]['other_rtplan']]
                if 'rts' in rt_dict[rt_tp].keys():
                    rts_dir = os.path.join(
                        out_dir, sub_name, session, 'RTSTRUCT')
                    os.makedirs(os.path.join(rts_dir, '1-RTSTRUCT_Used'),
                                exist_ok=True)
                    shutil.copy2(rt_dict[rt_tp]['rts'],
                                 os.path.join(rts_dir, '1-RTSTRUCT_Used'))
                    if rt_dict[rt_tp]['other_rts']:
                        other_dir = os.path.join(rts_dir, 'Other_RTSTRUCT')
                        os.makedirs(other_dir)
                        [shutil.copytree(x, os.path.join(other_dir, x.split('/')[-1]))
                         for x in rt_dict[rt_tp]['other_rts']]
                if 'rtct' in rt_dict[rt_tp].keys():
                    rtct_dir = os.path.join(
                        out_dir, sub_name, session, 'RTCT')
                    os.makedirs(os.path.join(rtct_dir, '1-BPLCT_Used'),
                                exist_ok=True)
                    for f in sorted(glob.glob(rt_dict[rt_tp]['rtct']+'/*')):
                        shutil.copy2(
                            f, os.path.join(rtct_dir, '1-BPLCT_Used'))
                    if rt_dict[rt_tp]['other_ct']:
                        other_dir = os.path.join(rtct_dir, 'Other_CT')
                        os.makedirs(other_dir)
                        [shutil.copytree(x, os.path.join(other_dir, x.split('/')[-1]))
                         for x in rt_dict[rt_tp]['other_ct']]
                if ('phy_dose' in rt_dict[rt_tp].keys() and
                        rt_dict[rt_tp]['phy_dose'] is not None):
                    phy_dose_dir = os.path.join(
                        out_dir, sub_name, session, 'RTDOSE',
                        rt_dict[rt_tp]['phy_dose'][0])
                    os.makedirs(phy_dose_dir, exist_ok=True)
                    shutil.copy2(rt_dict[rt_tp]['phy_dose'][1], phy_dose_dir)
                if ('rbe_dose' in rt_dict[rt_tp].keys() and
                        rt_dict[rt_tp]['rbe_dose'] is not None):
                    rbe_dose_dir = os.path.join(
                        out_dir, sub_name, session, 'RTDOSE',
                        rt_dict[rt_tp]['rbe_dose'][0])
                    os.makedirs(rbe_dose_dir, exist_ok=True)
                    shutil.copy2(rt_dict[rt_tp]['rbe_dose'][1], rbe_dose_dir)
                if ('other_rtdose' in rt_dict[rt_tp].keys() and
                        rt_dict[rt_tp]['other_rtdose']):
                    other_dir = os.path.join(
                        out_dir, sub_name, session, 'RTDOSE', 'Other_RTDOSE')
                    os.makedirs(other_dir)
                    [shutil.copytree(x, os.path.join(other_dir, x.split('/')[-1]))
                     for x in rt_dict[rt_tp]['other_rtdose']]
            if session_dict['RT']:
                self.session_labelling(session_dict, rert_max_time, mrrt_max_time_diff)
            else:
                toreprocess.append([session_dict['CT'], session_dict['MR']])
        if toreprocess:
            self.ct_labelling(toreprocess, out_dir, rert_max_time, mrrt_max_time_diff)
                        
                        
#             if not os.path.isdir(rt_dir):
#                 iflogger.info('No RT data found')
#                 rt_tocopy = []
#                 rt_sub_name = None
#             else:
#                 rt_sub_name = os.listdir(rt_dir)[0]
#                 rt_tocopy = sorted(glob.glob(os.path.join(rt_dir, rt_sub_name, '*')))
#             if sub_name is not None:
#                 if not os.path.isdir(os.path.join(out_dir, sub_name)):
#                     os.makedirs(os.path.join(out_dir, sub_name))
#                 for folder in rt_tocopy:
#                     folder_name = folder.split('/')[-1]
#                     shutil.copytree(folder, os.path.join(
#                         out_dir, sub_name, folder_name))
#                     if folder in mr_tocopy:
#                         session_dict['MR'].append([os.path.join(
#                             out_dir, sub_name, folder_name),
#                             dt.strptime(folder_name, '%Y%m%d')])
#                     elif folder in ct_tocopy:
#                         session_dict['CT'].append([os.path.join(
#                         out_dir, sub_name,folder_name),
#                         dt.strptime(folder_name, '%Y%m%d')])
#                     else:
#                         session_dict['RT'].append([os.path.join(
#                         out_dir, sub_name,folder_name),
#                         dt.strptime(folder_name.split('_RT')[0], '%Y%m%d')])
# 
# 
#         input_list = self.inputs.input_list
#         rert_max_time = self.inputs.rert_max_time
#         mrrt_max_time_diff = self.inputs.mrrt_max_time_diff
#         out_dir = os.path.abspath(self.inputs.out_folder)
# 
#         toreprocess = []
# 
#         for directories in input_list:
#             session_dict = {}
#             session_dict['RT'] = []
#             session_dict['CT'] = []
#             session_dict['MR'] = []
#             if len(directories) == 3:
#                 mr_dir = directories[2]
#                 rt_dir = directories[1]
#                 ct_dir = directories[0]
#             else:
#                 mr_dir = None
#                 rt_dir = directories[1]
#                 ct_dir = directories[0]
#             if mr_dir is None or not os.path.isdir(mr_dir):
#                 iflogger.info('No MRI data found')
#                 mr_tocopy = []
#                 mr_sub_name = None
#             else:
#                 mr_sub_name = os.listdir(mr_dir)[0]
#                 mr_tocopy = sorted(glob.glob(os.path.join(mr_dir, mr_sub_name, '*')))
#             if not os.path.isdir(rt_dir):
#                 iflogger.info('No RT data found')
#                 rt_tocopy = []
#                 rt_sub_name = None
#             else:
#                 rt_sub_name = os.listdir(rt_dir)[0]
#                 rt_tocopy = sorted(glob.glob(os.path.join(rt_dir, rt_sub_name, '*')))
#             if not os.path.isdir(ct_dir):
#                 iflogger.info('No CT data found')
#                 ct_tocopy = []
#                 ct_sub_name = None
#             else:
#                 ct_sub_name = os.listdir(ct_dir)[0]
#                 ct_tocopy = sorted(glob.glob(os.path.join(ct_dir, ct_sub_name, '*')))
# #             mr_sub_name = os.listdir(mr_dir)[0]
#             if (rt_sub_name is not None and mr_sub_name is not None) and rt_sub_name != mr_sub_name:
#                 raise Exception('Subject name is different between MR and RT '
#                                 'result folder. Something went wrong.')
#             if mr_sub_name is not None:
#                 sub_name = mr_sub_name
#             elif rt_sub_name is not None:
#                 sub_name = rt_sub_name
#             else:
#                 sub_name = None
# 
#             if sub_name is not None:
#                 if not os.path.isdir(os.path.join(out_dir, sub_name)):
#                     os.makedirs(os.path.join(out_dir, sub_name))
#                 for folder in mr_tocopy+rt_tocopy+ct_tocopy:
#                     folder_name = folder.split('/')[-1]
#                     shutil.copytree(folder, os.path.join(
#                         out_dir, sub_name, folder_name))
#                     if folder in mr_tocopy:
#                         session_dict['MR'].append([os.path.join(
#                             out_dir, sub_name, folder_name),
#                             dt.strptime(folder_name, '%Y%m%d')])
#                     elif folder in ct_tocopy:
#                         session_dict['CT'].append([os.path.join(
#                         out_dir, sub_name,folder_name),
#                         dt.strptime(folder_name, '%Y%m%d')])
#                     else:
#                         session_dict['RT'].append([os.path.join(
#                         out_dir, sub_name,folder_name),
#                         dt.strptime(folder_name.split('_RT')[0], '%Y%m%d')])
#             if session_dict['RT']:
#                 self.session_labelling(session_dict, rert_max_time, mrrt_max_time_diff)
#             else:
#                 toreprocess.append([session_dict['CT'], session_dict['MR']])
#         if toreprocess:
#             self.ct_labelling(toreprocess, out_dir, rert_max_time, mrrt_max_time_diff)

        return runtime

    def ct_labelling(self, toreprocess, out_dir, rert_max_time, mrrt_max_time_diff):

        rtct_sds = []
        for path, _, files in os.walk(out_dir):
            for f in files:
                if f == 'rtct_series_description.txt':
                    with open(os.path.join(path, f), 'r') as f:
                        rtct_sds.append(f.read())
#                         sds = [x.strip() for x in f.read()]
#                     rtct_sds =  rtct_sds + sds
        for tp in toreprocess:
            label = False
            session_dict = {}
            session_dict['RT'] = []
            session_dict['CT'] = []
            session_dict['MR'] = []
            ct_sessions, mr_sessions = tp
            for ct in sorted(ct_sessions):
                ct_path = ct[0]
                ct_dirs = sorted(glob.glob(ct_path+'/CT/*'))
                for ct_dir in ct_dirs:
                    sd = self.get_rtct_description(ct_dir, write2file=False)
                    if sd in rtct_sds:
                        ct_sessions.remove(ct)
                        ct_dirs.remove(ct_dir)
                        ct_folder_name = ct_dir.split('/')[-1]
                        new_name = (ct_dir.replace(ct_folder_name, '1-'+ct_folder_name)
                                    .replace('/CT/', '/RTCT/').replace('_CT', '_RT'))
                        shutil.move(ct_dir, new_name)
                        for d in ct_dirs:
                            if not os.path.isdir(new_name.split('/RTCT')[0]+'/Other_CT'):
                                os.makedirs(new_name.split('/RTCT')[0]+'/Other_CT')
                            shutil.move(d, new_name.split('/RTCT')[0]+'/Other_CT/')
                        session_dict['RT'].append([new_name.split('/RTCT')[0], ct[1]])
                        shutil.rmtree(ct_path)
                        label = True
                        break
            session_dict['CT'] = ct_sessions
            session_dict['MR'] = mr_sessions
            if label:
                self.session_labelling(session_dict, rert_max_time, mrrt_max_time_diff)
            else:
                iflogger.info('Could not identify any planning CT in the _CT folders '
                              'based on CT series descriptions.')
        
    def session_labelling(self, session_dict, rert_max_time, mrrt_max_time_diff):
        
        rt_sessions = sorted(session_dict['RT'])
        ct_sessions = sorted(session_dict['CT'])
        mr_sessions = sorted(session_dict['MR'])

        toremove = []
        if len(rt_sessions) > 1:
            rt_ref = rt_sessions[0]
            self.get_rtct_description(rt_ref[0]+'/RTCT/1-*')
            for i, rt_session in enumerate(rt_sessions[1:]):
                self.get_rtct_description(rt_session[0]+'/RTCT/1-*')
                diff = (rt_session[1]-rt_ref[1]).days
                if diff <= rert_max_time:
                    toremove.append(rt_session)
                    ct_sessions.append(rt_session)
                else:
                    rt_ref = rt_session
        else:
            self.get_rtct_description(rt_sessions[0][0]+'/RTCT/1-*')
        for f in toremove:
            rt_sessions.remove(f)
        
        mr_sessions_groups = []
        ct_sessions_groups = []
        if len(rt_sessions) > 1:
            for i in range(len(rt_sessions)-1):
                mr_sessions_groups.append(
                    [x for x in mr_sessions 
                     if x[1] <= (rt_sessions[i+1][1]-timedelta(days=30))])
                [mr_sessions.remove(x) for x in mr_sessions_groups[-1]]
                ct_sessions_groups.append(
                    [x for x in ct_sessions 
                     if x[1] <= (rt_sessions[i+1][1]-timedelta(days=30))])
                [ct_sessions.remove(x) for x in ct_sessions_groups[-1]]
            mr_sessions_groups.append(mr_sessions)
            ct_sessions_groups.append(ct_sessions)
        elif len(rt_sessions) == 1:
            mr_sessions_groups.append(mr_sessions)
            ct_sessions_groups.append(ct_sessions)
        
        sessions = {}
        for i, rt in enumerate(rt_sessions):
            rt_path, rt_date = rt
            basepath = '/'.join(rt_path.split('/')[:-1])
            basename = basepath+'_{}'.format(i+1)
            sessions[basename] = [rt_path]
#                 self.get_rtct_description(rt_path)
            mris = mr_sessions_groups[i]
            cts = ct_sessions_groups[i]
            if mris:
                mrtp = None
                diff = np.inf
                found = False
                for mr in mris:
                    mr_path, mr_date = mr
                    if mr_date <= rt_date and (rt_date - mr_date).days <= diff:
                        mrtp = mr
                        diff = (rt_date - mr_date).days
                        found = True
                    elif (mr_date > rt_date and (mr_date-rt_date).days <= mrrt_max_time_diff
                            and not found):
                        mrtp = mr
                if mrtp is not None:
                    shutil.move(mrtp[0], mrtp[0]+'_MR-RT')
                    mris.remove(mrtp)
                    sessions[basename].append(mrtp[0]+'_MR-RT')
                for mr in mris:
                    mr_path, mr_date = mr
                    if mr_date < rt_date:
                        shutil.move(mr_path, mr_path+'_pre-RT')
                        sessions[basename].append(mr_path+'_pre-RT')
                    elif mr_date > rt_date and mr_date <= rt_date+timedelta(days=rert_max_time):
                        shutil.move(mr_path, mr_path+'_post-RT')
                        sessions[basename].append(mr_path+'_post-RT')
                    elif mr_date > rt_date+timedelta(days=rert_max_time):
                        shutil.move(mr_path, mr_path+'_FU')
                        sessions[basename].append(mr_path+'_FU')
            if cts:
                for ct in cts:
                    ct_path, ct_date = ct
                    ct_outpath = ct_path.split('_CT')[0]
                    if ct_date < rt_date:
                        sessions[basename].append(ct_outpath+'_pre-RT')
                        try:
                            shutil.move(ct_path, ct_outpath+'_pre-RT')
                        except:
                            ff = sorted(glob.glob(ct_path+'/*'))
                            for f in ff:
                                fname = f.split('/')[-1]
                                shutil.move(f, ct_outpath+'_pre-RT'+'/{}'.format(fname))
                    elif ct_date > rt_date and ct_date <= rt_date+timedelta(days=rert_max_time):
                        sessions[basename].append(ct_outpath+'_post-RT')
                        try:
                            shutil.move(ct_path, ct_outpath+'_post-RT')
                        except:
                            ff = sorted(glob.glob(ct_path+'/*'))
                            for f in ff:
                                fname = f.split('/')[-1]
                                shutil.move(f, ct_outpath+'_post-RT'+'/{}'.format(fname))
                    elif ct_date > rt_date+timedelta(days=rert_max_time):
                        sessions[basename].append(ct_outpath+'_FU')
                        try:
                            shutil.move(ct_path, ct_outpath+'_FU')    
                        except:
                            ff = sorted(glob.glob(ct_path+'/*'))
                            for f in ff:
                                fname = f.split('/')[-1]
                                shutil.move(f, ct_outpath+'_FU'+'/{}'.format(fname))
                    elif ct_date == rt_date:
                        if os.path.isdir(rt_path+'/RTCT'):
                            shutil.rmtree(ct_path)
                        else:
                            ff = sorted(glob.glob(ct_path+'/*'))
                            for f in ff:
                                fname = f.split('/')[-1]
                                shutil.move(f, rt_path+'/{}'.format(fname))
                            shutil.rmtree(ct_path)
            sessions[basename] = list(set(sessions[basename]))
        if len(sessions) > 1:
            for key in sessions:
                if not os.path.isdir(key):
                    os.makedirs(key)
                for session in sessions[key]:
                    shutil.move(session, key)
            shutil.rmtree(basepath)
                

    def get_rtct_description(self, rt_dir, write2file=True):
        try:
            rtct = sorted(glob.glob(rt_dir+'/*.dcm'))[0]
            hd = pydicom.read_file(rtct)
            sd = hd.SeriesDescription
            sd = sd.lower().replace(' ', '').replace('.', '')
            if write2file:
                wdir = rt_dir.split('RTCT')[0]
                with open(wdir+'/rtct_series_description.txt', 'w') as f:
                    f.write(sd)
            else:
                return sd
        except:
            iflogger.info('No RTCT series description available.')

    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.out_folder):
            outputs['out_folder'] = os.path.abspath(
                self.inputs.out_folder)

        return outputs


class MHA2NIIConverterInputSpec(BaseInterfaceInputSpec):
    
    input_folder = Directory(help='Input directory to convert.')
    out_folder = Directory('Nifti_Data', usedefault=True,
                           desc='Folder with converted data.')


class MHA2NIIConverterOutputSpec(TraitedSpec):
    
    out_folder = Directory(help='Folder with converted data.')
    out_files = traits.List(help='List of converted files.')


class MHA2NIIConverter(BaseInterface):
    
    input_spec = MHA2NIIConverterInputSpec
    output_spec = MHA2NIIConverterOutputSpec
    
    def _run_interface(self, runtime):
        
        toconvert = sorted(glob.glob(
            os.path.join(self.inputs.input_folder, '*.mha')))
        self.converted_files = []
        if toconvert:
            for mha in toconvert:
                filename = mha.split('/')[-1].split('.mha')[0]
                outfile = filename+'.nii.gz'
                mha_file = sitk.ReadImage(mha)
                sitk.WriteImage(mha_file, outfile)
                self.converted_files.append(os.path.abspath(outfile))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.out_folder):
            outputs['out_folder'] = os.path.abspath(
                self.inputs.out_folder)
        outputs['out_files'] = self.converted_files

        return outputs


class SinkSortingInputSpec(BaseInterfaceInputSpec):
    
    tosink = Directory(help='Input directory to convert.')
    out_folder = Directory('Nifti_Data', usedefault=True,
                           desc='Folder with converted data.')


class SinkSortingOutputSpec(TraitedSpec):
    
    out_folder = Directory(help='Folder with converted data.')


class SinkSorting(BaseInterface):
    
    input_spec = SinkSortingInputSpec
    output_spec = SinkSortingOutputSpec
    
    def _run_interface(self, runtime):

        tosink = self.inputs.tosink
        out_folder = self.inputs.out_folder
        
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
        if os.path.isdir(tosink):
            shutil.move(tosink, out_folder)

        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.out_folder):
            outputs['out_folder'] = self.inputs.out_folder

        return outputs


class PreclinicalSinkInputSpec(BaseInterfaceInputSpec):
    
    tosink = traits.List(help='List of directories to sink.')
    out_folder = Directory('Sorted_data', usedefault=True,
                           desc='Folder with sorted data.')


class PreclinicalSinkOutputSpec(TraitedSpec):
    
    out_folder = Directory(help='Folder with converted data.')


class PreclinicalSink(BaseInterface):
    
    input_spec = PreclinicalSinkInputSpec
    output_spec = PreclinicalSinkOutputSpec
    
    def _run_interface(self, runtime):

        tosinks = self.inputs.tosink
        out_folder = os.path.join(self.inputs.out_folder, 'Sorted_data')
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
        
        for tosink in tosinks:
            subs = glob.glob(tosink+'/*')
            for sub in subs:
                sub_name = sub.split('/')[-1]
                result_dir = os.path.join(out_folder, sub_name)
                shutil.copytree(sub, result_dir)

        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.out_folder):
            outputs['out_folder'] = self.inputs.out_folder

        return outputs
