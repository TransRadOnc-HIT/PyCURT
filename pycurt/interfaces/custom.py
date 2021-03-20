import glob
import pydicom
import os
import shutil
import subprocess as sp
from collections import defaultdict
from nipype.interfaces.base import (
    BaseInterface, TraitedSpec, Directory,
    BaseInterfaceInputSpec, traits)
from nipype.interfaces.base import isdefined
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from pycurt.utils.torch import (
    resize_2Dimage, ZscoreNormalization, ToTensor,
    load_checkpoint, MRClassifierDataset)
from pycurt.utils.filemanip import create_move_toDir
from copy import deepcopy
import pickle


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

        return runtime

    def extract_plan(self, dir_name, out_dir):
    
        # FInding the RTplan which was used.( taking the last approved plan)
        # From the RTplan metadata, the structure and the doseCubes instance were taken
        if not os.path.isdir(dir_name):
            print('RT plan was not found. With no plan, the doseCubes, '
                  'struct, and planning CT instances cannot be extracted')
            return None, None, None
            
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
        if plan_name is None:
            return None,None,None

        ds = pydicom.dcmread(plan_name, force=True)
        try:
            rtstruct_instance = (ds.ReferencedStructureSetSequence[0]
                                 .ReferencedSOPInstanceUID)
        except:
            rtstruct_instance=None
        try:
            for i in range(0, len(ds.ReferencedDoseSequence)):
                singleDose_instance = (ds.ReferencedDoseSequence[i]
                                       .ReferencedSOPInstanceUID + '.dcm')
                dose_cubes_instance.append(singleDose_instance)
        except:
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
            return None
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
            return None

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
            return None

        dcm_files = glob.glob(dir_name+'/*/*.dcm')
        other_dose = []
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
                        rbe_name = None
                        rbe_dose = None
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
                        phy_dose = f
                        
                    else:
                        phy_dose = None
                        phy_name = None
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
    sub_checkpoints = traits.Dict(
        desc='Classification network weights for within modality inference '
        '(i.e. for T1 vs T1KM classification).')
    body_part = traits.List(['hnc', 'hncKM'], usdefault=True, desc=(
        'Body part of interest. If provided, only the images '
        'labeled as this key will be considered for sorting. '
        'This is only used for bp_class classification.'
        'Default is head and neck (hnc).'))
    network = traits.Enum('bpclass', 'mrclass', desc=(
        'Classification network to use for image classification. '
        'Possible values are: bpclass or mrclass.'))
    out_folder = Directory('Labelled_dir', usedefault=True,
                           desc='Labelled sorted folder.')


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
        sub_checkpoints = self.inputs.sub_checkpoints
        images2label = self.inputs.images2label
        output_dir = os.path.abspath(self.inputs.out_folder)
        body_part = self.inputs.body_part
        cl_network = self.inputs.network

        if cl_network == 'mrclass' and 'MR' in images2label.keys():
            mr_modality = {}
            mr_modality['MR'] = images2label['MR']
            images2label = mr_modality
        elif cl_network == 'mrclass':
            images2label = {}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        data_transforms = transforms.Compose(
            [resize_2Dimage(224), ZscoreNormalization(), ToTensor()])

        labeled_images = defaultdict()
        self.labelled_images = {}
        self.output_dict = {}
        for modality in images2label.keys():
            for_inference = images2label[modality]
#             #iteration through the different models
            labeled = defaultdict(list)
            for cl in checkpoints.keys():
                model, class_names, scan = load_checkpoint(checkpoints[cl])
                class_names[1] = '@lL'
                test_dataset = MRClassifierDataset(
                    list_images=for_inference, transform=data_transforms,
                    class_names=class_names, scan = scan)
                test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False,
                                             num_workers=1)
                for step, data in enumerate(test_dataloader):
                    inputs = data['image']
                    img_name = data['fn']
                    inputs = inputs.to(device)
                    output = model(inputs)
                    prob = output.data.cpu().numpy()
                    actRange = abs(prob[0][0])+abs(prob[0][1])
                    index = output.data.cpu().numpy().argmax()
                    if index == 0:
                        labeled[img_name[0]].append([cl, actRange])
                      
            # check double classification and compare the activation value of class 0
            for key in labeled.keys():
                r = 0
                j = 0
                for i in range(len(labeled[key])-1):
                    if labeled[key][i][1] > r:
                        r = labeled[key][i][1]
                        j = i
                labeled[key] = labeled[key].pop(j)
                  
            # check for the unlabeled images
            not_labeled = list(set(for_inference) - set(list(labeled.keys())))
            for img in not_labeled:
                labeled[img] = ['other', 0]
                  
            # prepare for subclassification   
            labeled_images[modality] = defaultdict(list)
            for key in labeled.keys():
                labeled_images[modality][labeled[key][0]].append(key)
                          
            # subclassification        
            labeled_sub = defaultdict(list)
                  
            for cl in sub_checkpoints.keys():
                model, class_names, scan = load_checkpoint(sub_checkpoints[cl])
                test_dataset = MRClassifierDataset(list_images = labeled_images[modality][cl], 
                                                   transform = data_transforms,
                                                   class_names = class_names, scan = scan, 
                                                   subclasses = True)
                test_dataloader = DataLoader(test_dataset, batch_size = 1,
                                             shuffle=False, num_workers=1)
                for step, data in enumerate(test_dataloader):
                    inputs = data['image']
                    img_name = data['fn']
                    inputs = inputs.to(device)
                    output = model(inputs)
                    prob = output.data.cpu().numpy()
                    actRange = abs(prob[0][0])+abs(prob[0][1])
                    index = output.data.cpu().numpy().argmax()
                    if index == 1:
                        c = 'KM'
                    else:
                        c = ''
                    labeled_sub[img_name[0]] = [cl+c, actRange]
                           
            for key in labeled_sub.keys():
                labeled[key] = labeled_sub[key]
                     
#             with open('/home/fsforazz/ww.pickle{}{}'.format(cl_network, modality), 'wb') as f:
#                 pickle.dump(labeled, f, protocol=pickle.HIGHEST_PROTOCOL)
#                 
#             with open('/home/fsforazz/ww.pickle{}{}'.format(cl_network, modality), 'rb') as handle:
#                 labeled = pickle.load(handle)
    
            labeled_images[modality] = defaultdict(list)
            for key in labeled.keys():
                labeled_images[modality][labeled[key][0]].append([key, labeled[key][1]])

            bps_of_interest = [x for x in labeled_images[modality].keys() if x in body_part]
            if cl_network == 'bpclass' and modality == 'MR':
                self.labelled_images[modality] = []
                for bp in bps_of_interest:
                    imgs = [x[0] for x in labeled_images[modality][bp]]
                    self.labelled_images[modality] = self.labelled_images[modality]+imgs
            elif cl_network == 'bpclass' and modality == 'CT':
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
                    self.output_dict[modality] = {}
                    if key != 'other':
                        self.output_dict[modality][key] = self.labelled_images[modality][key]
#                         for cm in self.labelled_images[modality][key]:
#                             indices = [i for i, x in enumerate(cm[0]) if x == "/"]
#                             if modality == 'MR':
#                                 dirName = os.path.join(
#                                     output_dir, cm[0][indices[-3]+1:indices[-1]], key)
#                             else:
#                                 dirName = os.path.join(
#                                     output_dir, cm[0][indices[-4]+1:indices[-1]])
#                             create_move_toDir(cm[0], dirName, cm[1])

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.out_folder):
            outputs['out_folder'] = os.path.abspath(
                self.inputs.out_folder)
            outputs['labeled_images'] = self.labelled_images
            outputs['output_dict'] =self.output_dict

        return outputs

