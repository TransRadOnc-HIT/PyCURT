import glob
import pydicom
import os
import nibabel as nib
import subprocess as sp
import numpy as np
from collections import defaultdict
from nipype.interfaces.base import (
    BaseInterface, TraitedSpec, Directory,
    BaseInterfaceInputSpec, traits, InputMultiPath)
from nipype.interfaces.base import isdefined
import torch
from torch.utils.data import DataLoader
from pycurt.utils.torch import (
    load_checkpoint, MRClassifierDataset_inference)
import nrrd
import cv2
from scipy.ndimage.interpolation import rotate
from scipy import ndimage
from skimage.measure import label, regionprops
from core.utils.filemanip import split_filename
import matplotlib.pyplot as plot
from pycurt.classifier.inference import run_inference_bpclass,\
    run_inference_mrclass


ExplicitVRLittleEndian = '1.2.840.10008.1.2.1'
ImplicitVRLittleEndian = '1.2.840.10008.1.2'
DeflatedExplicitVRLittleEndian = '1.2.840.10008.1.2.1.99'
ExplicitVRBigEndian = '1.2.840.10008.1.2.2'
NotCompressedPixelTransferSyntaxes = [ExplicitVRLittleEndian,
                                      ImplicitVRLittleEndian,
                                      DeflatedExplicitVRLittleEndian,
                                      ExplicitVRBigEndian]


RESOURCES_PATH = os.path.abspath(os.path.join(os.path.split(__file__)[0],
                                 os.pardir, os.pardir, 'resources'))

class RTDataSortingInputSpec(BaseInterfaceInputSpec):
    
    input_dir = Directory(exists=True, help='Input directory to sort.')
    out_folder = Directory('RT_sorted_dir', usedefault=True,
                           desc='RT data sorted folder.')


class RTDataSortingOutputSpec(TraitedSpec):
    
    out_folder = Directory(help='RT Sorted folder.')
    output_dict = traits.Dict()


class RTDataSorting(BaseInterface):
    
    input_spec = RTDataSortingInputSpec
    output_spec = RTDataSortingOutputSpec
    
    def _run_interface(self, runtime):

        input_dir = self.inputs.input_dir
        out_dir = os.path.abspath(self.inputs.out_folder)

        modality_list = [ 'RTPLAN' , 'RTSTRUCT', 'RTDOSE', 'CT']
        
        input_tp_folder = list(set([x for x in glob.glob(input_dir+'/*/*')
                                    for y in glob.glob(x+'/*')
                                    for r in modality_list if r in y]))

        self.output_dict = {}
        for tp_folder in input_tp_folder:
            sub_name, tp = tp_folder.split('/')[-2:]
            key_name = sub_name+'_'+tp
            self.output_dict[key_name] = {}
            out_basedir = os.path.join(out_dir, sub_name, tp+'_RT')
            print('Processing Sub: {0}, timepoint: {1}'.format(sub_name, tp))

            plan_name, rtstruct_instance, dose_cubes_instance, ot_plans = self.extract_plan(
                os.path.join(tp_folder, 'RTPLAN'), os.path.join(out_basedir, 'RTPLAN'))
            if plan_name is None:
                continue
            else:
                self.output_dict[key_name]['rtplan'] = plan_name
                self.output_dict[key_name]['other_rtplan'] = ot_plans

            if rtstruct_instance is not None:
                ct_classInstance, rts, ot_rts = self.extract_struct(
                    os.path.join(tp_folder, 'RTSTRUCT'), rtstruct_instance,
                    os.path.join(out_basedir, 'RTSTRUCT'))
                self.output_dict[key_name]['rts'] = rts
                self.output_dict[key_name]['other_rts'] = ot_rts
            else:
                print('The RTSTRUCT was not found. With no RTSTRUCT, '
                      'the planning CT instances cannot be extracted')
                ct_classInstance = None
            if ct_classInstance is not None:
                rtct, ot_rtct = self.extract_BPLCT(
                    os.path.join(tp_folder, 'CT'), ct_classInstance,
                    os.path.join(out_basedir, 'RTCT'))
                self.output_dict[key_name]['rtct'] = rtct
                self.output_dict[key_name]['other_ct'] = ot_rtct
            if dose_cubes_instance is not None:
                phy_d, phy_n, rbe_d, rbe_n, ot_d = self.extract_dose_cubes(
                    os.path.join(tp_folder, 'RTDOSE'), dose_cubes_instance,
                    os.path.join(out_basedir, 'RTDOSE'))
                if phy_d is not None:
                    self.output_dict[key_name]['phy_dose'] = [phy_n, phy_d]
                else:
                    self.output_dict[key_name]['phy_dose'] = None
                if rbe_d is not None:
                    self.output_dict[key_name]['rbe_dose'] = [
                        os.path.join(out_basedir, 'RTDOSE', rbe_n),
                        rbe_d]
                else:
                    self.output_dict[key_name]['rbe_dose'] = None
                self.output_dict[key_name]['other_rtdose'] = ot_d
        d = dict([(k, v) for k, v in self.output_dict.items() if v])
        self.output_dict = d

        return runtime

    def extract_plan(self, dir_name, out_dir):
    
        # FInding the RTplan which was used.( taking the last approved plan)
        # From the RTplan metadata, the structure and the doseCubes instance were taken
        if not os.path.isdir(dir_name):
            print('RT plan was not found. With no plan, the doseCubes, '
                  'struct, and planning CT instances cannot be extracted')
            return None, None, None, None
            
        plan_date, plan_time = 0, 0
        dose_cubes_instance = []
        plan_name = None
        radiation_type = defaultdict(list)

        dcm_files = glob.glob(dir_name+'/*/*.dcm')
    
        # check if multiple radiation treatment has been given
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(f, force=True)
            except:
                continue
            if hasattr(ds, 'BeamSequence'):
                rt = ds.BeamSequence[0].RadiationType
            elif hasattr(ds, 'IonBeamSequence'):
                rt = ds.IonBeamSequence[0].RadiationType
            radiation_type[rt].append(f)

        for f in dcm_files:
            try:
                ds = pydicom.dcmread(f, force=True)
            except:
                continue
            # check if RT plan has plan intent attribute and approval status
                # .If no, default taken as curative and approved
            if hasattr(ds, 'ApprovalStatus'):
                status_check = ds.ApprovalStatus
            else:
                status_check = 'APPROVED'
            if hasattr(ds, 'PlanIntent '):
                plan_intent_check = ds.PlanIntent
            else:
                plan_intent_check = 'CURATIVE'
            if status_check == 'APPROVED' and plan_intent_check == 'CURATIVE':
                plan_curr_plan_date = float(ds.RTPlanDate)
                plan_curr_plan_time = float(ds.RTPlanTime)
                if plan_curr_plan_date > plan_date:
                    plan_date = plan_curr_plan_date
                    plan_time = plan_curr_plan_time
                    plan_name = f
                elif plan_curr_plan_date == plan_date:
                    if plan_curr_plan_time > plan_time:
                        plan_date = plan_curr_plan_date
                        plan_time = plan_curr_plan_time
                        plan_name = f
        if plan_name is None and len(dcm_files) == 1:
            plan_name = dcm_files[0]
        elif plan_name is None and len(dcm_files) != 1: 
            return None, None, None, None

        ds = pydicom.dcmread(plan_name, force=True)
        try:
            rtstruct_instance = (ds.ReferencedStructureSetSequence[0]
                                 .ReferencedSOPInstanceUID)
        except:
            rtstruct_instance=None
        try:
            dose_seq = ds.ReferencedDoseSequence
        except AttributeError:
            try:
                dose_seq = ds.DoseReferenceSequence
                for i in range(0, len(dose_seq)):
                    singleDose_instance = (ds.ReferencedDoseSequence[i]
                                           .ReferencedSOPInstanceUID + '.dcm')
                    dose_cubes_instance.append(singleDose_instance)
            except AttributeError:
                dose_cubes_instance = None

        plan_dir_old = os.path.split(plan_name)[0]
        plan_dir = os.path.join(out_dir, '1-RTPLAN_Used')
#         os.makedirs(plan_dir)
#         shutil.copy2(plan_name, plan_dir)
        other_plan = [x for x in glob.glob(dir_name+'/*') if x != plan_dir_old]
#         if other_plan:
#             other_dir = os.path.join(out_dir, 'Other_RTPLAN')
#             os.makedirs(other_dir)
#             [shutil.copytree(x, os.path.join(other_dir, x.split('/')[-1]))
#              for x in other_plan]

        return plan_name, rtstruct_instance, dose_cubes_instance, other_plan

    def extract_struct(self, dir_name, rtstruct_instance, out_dir):
        # FInding the RTstruct which was used.( based on the RTsrtuct reference instance in
        # the RTplan metadata)
        ct_class_instance = None
        if not os.path.exists(dir_name) and not os.path.isdir(dir_name):
            print('RTStruct was not found..')
            return None, None, None
        dcm_files=glob.glob(dir_name+'/*/*.dcm')
        for f in dcm_files:
            ds = pydicom.dcmread(f,force=True)
            if ds.SOPInstanceUID == rtstruct_instance:
                try:
                    ct_class_instance = ds.ReferencedFrameOfReferenceSequence[0] \
                    .RTReferencedStudySequence[0].RTReferencedSeriesSequence[0] \
                    .SeriesInstanceUID
                except:
                    ct_class_instance = None          
                struct_dir = os.path.join(out_dir, '1-RTSTRUCT_Used')
#                 os.makedirs(struct_dir)
#                 shutil.copy2(f, struct_dir)
                break
        struct_old_dir = os.path.split(f)[0]
        other_rt = [x for x in glob.glob(dir_name+'/*') if x != struct_old_dir]
#         if other_rt:
#             other_dir = os.path.join(out_dir, 'Other_RTSTRUCT')
#             os.makedirs(other_dir)
#             [shutil.copytree(x, os.path.join(other_dir, x.split('/')[-1]))
#              for x in other_rt]

        return ct_class_instance, f, other_rt

    def extract_BPLCT(self, dir_name, ct_class_instance, out_dir):

        if not os.path.exists(dir_name) and not os.path.isdir(dir_name):
            print('BPLCT was not found..')
            return None, None

        dcm_folders = glob.glob(dir_name+'/*')
        dcm_folders = [x for x in dcm_folders if os.path.isdir(x)]
        for image in dcm_folders:
            img_name = image.split('/')[-1]
            dcm_files=[os.path.join(image, item) for item in os.listdir(image)
                       if ('.dcm' in item)]
            try:
                ds = pydicom.dcmread(dcm_files[0],force=True)
                series_instance_uid = ds.SeriesInstanceUID
            except:
                series_instance_uid = ''
            if  series_instance_uid == ct_class_instance:
                BPLCT_dir = os.path.join(out_dir, '1-BPLCT_Used_'+img_name)
#                 os.makedirs(BPLCT_dir)
#                 for f in dcm_files:
#                     shutil.copy2(f, BPLCT_dir)
                break
        ct_old_dir = os.path.split(dcm_files[0])[0]
        other_ct = [x for x in glob.glob(dir_name+'/*') if x != ct_old_dir
                    and os.path.isdir(x)]
        if other_ct:
            other_dir = os.path.join(out_dir, 'Other_CT')
#             os.makedirs(other_dir)
#             [shutil.copytree(x, os.path.join(other_dir, x.split('/')[-1]))
#              for x in other_ct]
        return image, other_ct

    def extract_dose_cubes(self, dir_name, dose_cubes_instance, out_dir):

        dose_physical_found = False
        dose_rbe_found = False
        if not os.path.isdir(dir_name):
            print('RTDOSE was not found..')
            return None, None, None, None, None

        dcm_files = glob.glob(dir_name+'/*/*.dcm')
        other_dose = []
        phy_dose = None
        phy_name = None
        rbe_name = None
        rbe_dose = None
        for f in dcm_files:
#             indices = [i for i, x in enumerate(f) if x == "/"]
            folder_name, f_name = f.split('/')[-2:]
            if all(f_name != dose_cubes_instance[i] \
                   for i in range(0, len(dose_cubes_instance))) and dose_cubes_instance!="":
#             if all(f[indices[-1]+1:] != dose_cubes_instance[i] \
#                    for i in range(0, len(dose_cubes_instance))) and dose_cubes_instance!="":

                other_dir = os.path.join(out_dir, 'Other_RTDOSE', folder_name)
#                 if not os.path.isdir(other_dir):
#                     os.makedirs(other_dir)
#                 shutil.copy2(f, other_dir)
                other_dose.append(f)
#                 if not os.listdir(f[0:indices[-1]]):
#                     os.rmdir(f[0:indices[-1]])
            else:
                try:
                    ds = pydicom.dcmread(f,force=True)
                    dose_type = ds.DoseType
                    dose_summation_type = ds.DoseSummationType
                except:
                    dose_type = ''
                    dose_summation_type = ''
                #check whether the dose is compressed, if yes decompress
                if ds.file_meta.TransferSyntaxUID not in \
                        NotCompressedPixelTransferSyntaxes:
                    self.decompress_dose(f)
                if dose_type == 'EFFECTIVE':
                    if 'PLAN' in dose_summation_type:
                        rbe_name = '1-RBE_Used'
                        dose_rbe_found = True
                    elif dose_summation_type == 'FRACTION':
                        rbe_name = '1-RBEFRACTION_Used'
                        dose_rbe_found = True
                    if dose_rbe_found:
                        rbe_dir = os.path.join(out_dir, rbe_name)
#                         if not os.path.isdir(rbe_dir):
#                             os.makedirs(rbe_dir)
#                         shutil.copy2(f, rbe_dir)
                        rbe_dose = f
                        
                    else:
                        print('dose_RBE_Cube was not found.')
                if dose_type == 'PHYSICAL':
                    if 'PLAN' in dose_summation_type:
                        phy_name = '1-PHYSICAL_Used'
                        dose_physical_found=True
                    elif dose_summation_type == 'FRACTION':
                        phy_name = '1-PHYSICALFRACTION_Used'
                        dose_physical_found=True
                    if dose_physical_found:
                        phy_dir = os.path.join(out_dir, phy_name)
#                         if not os.path.isdir(phy_dir):
#                             os.makedirs(phy_dir)
#                         shutil.copy2(f, phy_dir)
                    else:
                        phy_dose = f
                        
                        print('dose_Physical_Cube was not found.')
        return phy_dose, phy_name, rbe_dose, rbe_name, other_dose

    def decompress_dose(self, i):

        cmd = ("dcmdjpeg {0} {1} ".format(i, i))
        sp.check_output(cmd, shell=True)

    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.out_folder):
            outputs['out_folder'] = os.path.abspath(
                self.inputs.out_folder)
            outputs['output_dict'] = self.output_dict

        return outputs


class ImageClassificationInputSpec(BaseInterfaceInputSpec):
    
    images2label = traits.Dict(desc='List of images to be labelled.')
    checkpoints = traits.Dict(desc='Classification network weights.')
    modality = traits.Str(desc='Image modality ("MR" or "CT").')
    sub_checkpoints = traits.Dict(
        desc='Classification network weights for within modality inference '
        '(i.e. for T1 vs T1KM classification).')
    body_part = traits.List(['hnc'], usdefault=True, desc=(
        'Body part of interest. If provided, only the images '
        'labeled as this key will be considered for sorting. '
        'This is only used for bp_class classification.'
        'Default is head and neck (hnc).'))
    network = traits.Enum('bpclass', 'mrclass', desc=(
        'Classification network to use for image classification. '
        'Possible values are: bpclass or mrclass.'))
    out_folder = Directory('Labelled_dir', usedefault=True,
                           desc='Labelled sorted folder.')
    probability_th = traits.Float(desc='Only images classified as the body part '
                                  'of interest that have a probability higher '
                                  'than this values will be selected.')


class ImageClassificationOutputSpec(TraitedSpec):

    out_folder = Directory(help='Labelled folder.')
    labeled_images = traits.Dict(
        help='Dictionary with all labeled images')
    output_dict = traits.Dict(
        help='Dictionary with the labeled images to sink')


class ImageClassification(BaseInterface):
    
    input_spec = ImageClassificationInputSpec
    output_spec = ImageClassificationOutputSpec
    
    def _run_interface(self, runtime):
        
        checkpoints = self.inputs.checkpoints
        images2label = self.inputs.images2label
        body_part = self.inputs.body_part
        cl_network = self.inputs.network
        modality = self.inputs.modality
        sub_checkpoints = self.inputs.sub_checkpoints
        probability_th = self.inputs.probability_th

        labeled_images = defaultdict()
        self.labelled_images = {}
        self.output_dict = {}
        for modality in images2label.keys():
            self.output_dict[modality] = {}
            for_inference = images2label[modality]
            if cl_network == 'bpclass':
                labeled = run_inference_bpclass(
                    for_inference, checkpoints, modality=modality.lower(),
                    body_parts=body_part, th=probability_th)
            else:
                labeled = run_inference_mrclass(
                    for_inference, checkpoints, sub_checkpoints)

#             with open('/home/fsforazz/ww.pickle{}{}'.format(cl_network, modality), 'wb') as f:
#                 pickle.dump(labeled, f, protocol=pickle.HIGHEST_PROTOCOL)
#                    
#             with open('/home/fsforazz/ww.pickle{}{}'.format(cl_network, modality), 'rb') as handle:
#                 labeled = pickle.load(handle)
    
            labeled_images[modality] = defaultdict(list)
            for key in labeled.keys():
                labeled_images[modality][labeled[key][0]].append([key, labeled[key][1]])

            bps_of_interest = [x for x in labeled_images[modality].keys() if x in body_part]
            tmp_labelled = {}
            if cl_network == 'bpclass' and modality.lower() == 'mr':
                self.labelled_images[modality] = []
                tmp_labelled[modality] = {}
                for bp in bps_of_interest:
                    tmp_labelled[modality][bp] = labeled_images[modality][bp]
                    imgs = [x[0] for x in labeled_images[modality][bp]]
                    self.labelled_images[modality] = self.labelled_images[modality]+imgs
            elif cl_network == 'bpclass' and modality.lower() == 'ct':
                self.labelled_images[modality] = {}
                for bp in bps_of_interest:
                    self.labelled_images[modality][bp] = labeled_images[modality][bp]
            else:
                self.labelled_images[modality] = labeled_images[modality]
    
            to_remove = []
    
            for i in for_inference:
                image_dir = '/'.join(i.split('/')[:-1])
                to_remove = to_remove + [x for x in glob.glob(image_dir+'/*')
                                         if '.json' in x  or '.bval' in x
                                         or '.bvec' in x]
    
            for f in to_remove:
                if os.path.isfile(f):
                    os.remove(f)
                    
            if ((cl_network == 'bpclass' and modality == 'CT') or 
                    (modality == 'MR' and cl_network == 'mrclass')):
                for key in self.labelled_images[modality].keys():
                    if key != 'other':
                        self.output_dict[modality][key] = self.labelled_images[modality][key]
            elif cl_network == 'bpclass' and modality == 'MR':
                for key in tmp_labelled[modality].keys():
                    if key != 'other':
                        self.output_dict[modality][key] = tmp_labelled[modality][key]
            else:
                self.output_dict[modality] = None

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.out_folder):
            outputs['out_folder'] = os.path.abspath(
                self.inputs.out_folder)
            outputs['labeled_images'] = self.labelled_images
            outputs['output_dict'] = self.output_dict

        return outputs


class MouseCroppingInputSpec(BaseInterfaceInputSpec):
    
    ct = InputMultiPath(traits.File(exists=True), desc='Mouse clinical CT image to crop')
    out_folder = Directory('Cropping_dir', usedefault=True,
                           desc='Folder to store the cropping results.')


class MouseCroppingOutputSpec(TraitedSpec):
    
    cropped_dir = Directory(desc='Directory with all the cropped images.')
    cropped_images = traits.List()


class MouseCropping(BaseInterface):

    input_spec = MouseCroppingInputSpec
    output_spec = MouseCroppingOutputSpec
    
    def _run_interface(self, runtime):

        images = self.inputs.ct
        base_output_dir = os.path.abspath(self.inputs.out_folder)
        for image in images:
            sub_name, session, _, im_name = image.split('/')[-4:]
            base_outname = im_name.split('-')[0]
            output_dir = os.path.join(base_output_dir, sub_name, session)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            _, _, extention = split_filename(image)
            if extention == '.nii.gz' or extention == '.nii':
                ref = nib.load(image)
                ref = nib.as_closest_canonical(ref)
                image_hd = ref.header
                space_x, space_y, space_z = image_hd.get_zooms()
                im = ref.get_fdata()
            elif extention == '.nrrd':
                im, image_hd = nrrd.read(image)
                space_x = np.abs(image_hd['space directions'][0, 0])
                space_y = np.abs(image_hd['space directions'][1, 1])
                space_z = np.abs(image_hd['space directions'][2, 2])
            process = True
            out = []
    
            min_size_x = int(17 / space_x)
            if min_size_x > im.shape[0]:
                min_size_x = im.shape[0]
            min_size_y = int(30 / space_y)
            if min_size_y > im.shape[1]:
                min_size_y = im.shape[1]
            min_size_z = int(60 / space_z)
            if min_size_z > im.shape[2]:
                min_size_z = im.shape[2]
    
            _, _, dimZ = im.shape
    
            mean_Z = int(np.ceil((dimZ)/2))
            n_mice_detected = []
            not_correct = True
            angle = 0
            counter = 0
            while not_correct:
                im[im<np.min(im)+824] = np.min(im)
                im[im == 0] = np.min(im)
                for offset in [20, 10, 0, -10, -20]:
                    _, y1 = np.where(im[:, :, mean_Z+offset] != np.min(im))
                    im[:, np.min(y1)+min_size_y+10:, mean_Z+offset] = 0
                    img2, _, _ = self.find_cluster(im[:, :, mean_Z+offset], space_x)
                    labels = label(img2)
                    regions = regionprops(labels)
                    if regions:
                        n_mice_detected.append(len(regions))
                        if offset == 0:
                            xx = [x for y in [[x.bbox[0], x.bbox[2]] for x in regions] for x in y]
                            yy = [x for y in [[x.bbox[1], x.bbox[3]] for x in regions] for x in y]
                    else:
                        n_mice_detected.append(0)
                if len(set(n_mice_detected)) == 1 or (len(set(n_mice_detected)) == 2 and 0 in set(n_mice_detected)):
                    not_correct = False
                elif counter < 8:
                    angle = angle - 2
                    print('Different number of mice have been detected going from down-up '
                                   'in the image. This might be due to an oblique orientation '
                                   'of the mouse trail. The CT image will be rotated about the z '
                                   'direction of %f degrees', np.abs(angle))
                    n_mice_detected = []
                    if extention == '.nii.gz' or extention == '.nii':
                        im = nib.load(image)
                        im = nib.as_closest_canonical(im)
                        im = im.get_fdata()
                    elif extention == '.nrrd':
                        im, _ = nrrd.read(image)
                    im = rotate(im, angle, (0, 2), reshape=False, order=0)
                    counter += 1
                    if counter % 2 == 0:
                        mean_Z = mean_Z - 10
                else:
                    print('CT image has been rotated of 14° but the number of mice detected '
                                   'is still not the same going from down to up. This CT cannot be '
                                   'cropped properly and will be excluded.')
                    process = False
                    not_correct = False
    
            if process:
                if extention == '.nii.gz' or extention == '.nii':
                    im = nib.load(image)
                    im = nib.as_closest_canonical(im)
                    im = im.get_fdata()
                elif extention == '.nrrd':
                    im, _ = nrrd.read(image)
                if angle != 0:
                    im = rotate(im, angle, (0, 2), reshape=False, order=0)
                    im[im == 0] = np.min(im)
                im[im<np.min(im)+824] = np.min(im)
                im[im == 0] = np.min(im)
                im = im[xx[0]:xx[1], yy[0]:yy[1], :]
                hole_size = np.zeros(im.shape[2])
                offset_z = int((im.shape[2]-min_size_z)/2)
                for z in range(offset_z, im.shape[2]-offset_z):
                    _, _, zeros = self.find_cluster(im[:, :, z], space_x)
                    hole_size[z] = zeros
                mean_Z = np.where(hole_size==np.max(hole_size))[0][0]
                if extention == '.nii.gz' or extention == '.nii':
                    im = nib.load(image)
                    im = nib.as_closest_canonical(im)
                    im = im.get_fdata()
                elif extention == '.nrrd':
                    im, _ = nrrd.read(image)
                if angle != 0:
                    im = rotate(im, angle, (0, 2), reshape=False, order=0)
                    im[im == 0] = np.min(im)
                im[im<np.min(im)+824] = np.min(im)
                im[im == 0] = np.min(im)
    
                _, y1 = np.where(im[:, :, mean_Z] != np.min(im))
                im[:, np.min(y1)+min_size_y+10:, mean_Z] = 0
                img2, _, _ = self.find_cluster(im[:, :, mean_Z], space_x)
                labels = label(img2)
                regions = regionprops(labels)
                xx = [x for y in [[x.bbox[0], x.bbox[2]] for x in regions] for x in y]
                yy = [x for y in [[x.bbox[1], x.bbox[3]] for x in regions] for x in y]
    
                if extention == '.nii.gz' or extention == '.nii':
                    im = nib.load(image)
                    im = nib.as_closest_canonical(im)
                    im = im.get_fdata()
                elif extention == '.nrrd':
                    im, _ = nrrd.read(image)
                if angle != 0:
                    im = rotate(im, angle, (0, 2), reshape=False, order=0)
                    im[im == 0] = np.min(im)
    
                average_mouse_size = int(np.round(np.mean([xx[i+1]-xx[i] for i in range(0, len(xx), 2)])))
    
                average_hole_size = average_mouse_size // 2
                
                image_names = ['mouse-0{}'.format(x+1) for x in range(int(len(xx)//2))]
    
                offset_box = average_hole_size // 3
                y_min = np.min(yy) - offset_box
                y_max = np.max(yy) + offset_box
                for n_mice, i in enumerate(range(0, len(xx), 2)):
                    croppedImage = im[xx[i]-offset_box:xx[i+1]+offset_box, y_min:y_max,
                                      mean_Z-int(min_size_z/2):mean_Z+int(min_size_z/2)]
    
                    outname = os.path.join(
                        output_dir, base_outname+'-{}{}'.format(image_names[n_mice], extention))
                    if extention == '.nii.gz' or extention == '.nii':
                        im2save = nib.Nifti1Image(croppedImage, affine=ref.affine)
                        nib.save(im2save, outname)
                    elif extention == '.nrrd':
                        nrrd.write(outname, croppedImage, header=image_hd)
                    out.append(outname)

        self.cropped_images = out

        return runtime

    def find_cluster(self, im, spacing):

        im[im == np.min(im)] = 0
        im[im != 0] = 1

        nb_components, output, stats, _ = (
            cv2.connectedComponentsWithStats(im.astype(np.uint8),
                                             connectivity=8))
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        min_size = 100/spacing
        img2 = np.zeros((output.shape))
        cluster_size = []
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                cluster_size.append(sizes[i])
                img2[output == i + 1] = 1
        img2_filled = ndimage.binary_fill_holes(img2)
        zeros = np.sum(img2_filled-img2)

        return img2, cluster_size, zeros

    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.out_folder):
            outputs['cropped_dir'] = os.path.abspath(
                self.inputs.out_folder)
            outputs['cropped_images'] = self.cropped_images

        return outputs
