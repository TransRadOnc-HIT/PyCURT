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
    load_checkpoint, MRClassifierDataset_test)
from pycurt.utils.filemanip import create_move_toDir


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

        for tp_folder in input_tp_folder:
            sub_name, tp = tp_folder.split('/')[-2:]
            out_basedir = os.path.join(out_dir, sub_name, tp+'_RT')
            print('Processing Sub: {0}, timepoint: {1}'.format(sub_name, tp))

            plan_name, rtstruct_instance, dose_cubes_instance = self.extract_plan(
                os.path.join(tp_folder, 'RTPLAN'), os.path.join(out_basedir, 'RTPLAN'))
            if plan_name is None:
#                 out_basedir = os.path.join(out_dir, sub_name, 'CT_'+tp)
                out_basedir = os.path.join(out_dir, sub_name, tp+'_CT')
                if not os.path.isdir(out_basedir+'/CT'):
                    if os.path.isdir(tp_folder+'/CT'):
                        shutil.copytree(tp_folder+'/CT', out_basedir+'/CT')
#                 session_dict['CT'].append([out_basedir, dt.strptime(tp, '%Y%m%d')])
                continue
#             else:
#                 session_dict['RT'].append([out_basedir, dt.strptime(tp, '%Y%m%d')])
            if rtstruct_instance is not None:
                ct_classInstance = self.extract_struct(os.path.join(tp_folder, 'RTSTRUCT'),
                                                       rtstruct_instance,
                                                       os.path.join(out_basedir, 'RTSTRUCT'))
            else:
                print('The RTSTRUCT was not found. With no RTSTRUCT, '
                      'the planning CT instances cannot be extracted')
                ct_classInstance = None
            if ct_classInstance is not None:
                self.extract_BPLCT(os.path.join(tp_folder, 'CT'), ct_classInstance,
                                   os.path.join(out_basedir, 'RTCT'))
            if dose_cubes_instance is not None:
                self.extract_dose_cubes(os.path.join(tp_folder, 'RTDOSE'), dose_cubes_instance,
                                        os.path.join(out_basedir, 'RTDOSE'))

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
        os.makedirs(plan_dir)
        shutil.copy2(plan_name, plan_dir)
        other_plan = [x for x in glob.glob(dir_name+'/*') if x != plan_dir_old]
        if other_plan:
            other_dir = os.path.join(out_dir, 'Other_RTPLAN')
            os.makedirs(other_dir)
            [shutil.copytree(x, os.path.join(other_dir, x.split('/')[-1]))
             for x in other_plan]

        return plan_name, rtstruct_instance, dose_cubes_instance

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
                os.makedirs(struct_dir)
                shutil.copy2(f, struct_dir)
                break
        struct_old_dir = os.path.split(f)[0]
        other_rt = [x for x in glob.glob(dir_name+'/*') if x != struct_old_dir]
        if other_rt:
            other_dir = os.path.join(out_dir, 'Other_RTSTRUCT')
            os.makedirs(other_dir)
            [shutil.copytree(x, os.path.join(other_dir, x.split('/')[-1]))
             for x in other_rt]

        return ct_class_instance

    def extract_BPLCT(self, dir_name, ct_class_instance, out_dir):

        if not os.path.exists(dir_name) and not os.path.isdir(dir_name):
            print('BPLCT was not found..')
            return None

        dcm_folders = glob.glob(dir_name+'/*')
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
                os.makedirs(BPLCT_dir)
                for f in dcm_files:
                    shutil.copy2(f, BPLCT_dir)
                break
        ct_old_dir = os.path.split(f)[0]
        other_ct = [x for x in glob.glob(dir_name+'/*') if x != ct_old_dir]
        if other_ct:
            other_dir = os.path.join(out_dir, 'Other_CT')
            os.makedirs(other_dir)
            [shutil.copytree(x, os.path.join(other_dir, x.split('/')[-1]))
             for x in other_ct]

    def extract_dose_cubes(self, dir_name, dose_cubes_instance, out_dir):

        dose_physical_found = False
        dose_rbe_found = False
        if not os.path.isdir(dir_name):
            print('RTDOSE was not found..')
            return None

        dcm_files = glob.glob(dir_name+'/*/*.dcm')

        for f in dcm_files:
#             indices = [i for i, x in enumerate(f) if x == "/"]
            folder_name, f_name = f.split('/')[-2:]
            if all(f_name != dose_cubes_instance[i] \
                   for i in range(0, len(dose_cubes_instance))) and dose_cubes_instance!="":
#             if all(f[indices[-1]+1:] != dose_cubes_instance[i] \
#                    for i in range(0, len(dose_cubes_instance))) and dose_cubes_instance!="":

                other_dir = os.path.join(out_dir, 'Other_RTDOSE', folder_name)
                if not os.path.isdir(other_dir):
                    os.makedirs(other_dir)
                shutil.copy2(f, other_dir)
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
                        if not os.path.isdir(rbe_dir):
                            os.makedirs(rbe_dir)
                        shutil.copy2(f, rbe_dir)
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
                        if not os.path.isdir(phy_dir):
                            os.makedirs(phy_dir)
                        shutil.copy2(f, phy_dir)
                    else:
                        print('dose_Physical_Cube was not found.')

    def decompress_dose(self, i):

        cmd = ("dcmdjpeg {0} {1} ".format(i, i))
        sp.check_output(cmd, shell=True)

    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.out_folder):
            outputs['out_folder'] = os.path.abspath(
                self.inputs.out_folder)

        return outputs


class MRClassInputSpec(BaseInterfaceInputSpec):
    
    mr_images = traits.List(desc='List of MR images to be labelled.')
    checkpoints = traits.Dict(desc='MRClass weights.')
    sub_checkpoints = traits.Dict(
        desc='MRClass weights for within modality inference '
        '(i.e. for T1 vs T1KM classification).')
    out_folder = Directory('MR_sorted_dir', usedefault=True,
                           desc='MR data sorted folder.')

class MRClassOutputSpec(TraitedSpec):

    out_folder = Directory(help='MR Sorted folder.')


class MRClass(BaseInterface):
    
    input_spec = MRClassInputSpec
    output_spec = MRClassOutputSpec
    
    def _run_interface(self, runtime):
        
        checkpoints = self.inputs.checkpoints
        sub_checkpoints = self.inputs.sub_checkpoints
        for_inference = self.inputs.mr_images
        output_dir = os.path.abspath(self.inputs.out_folder)

        th = {'ADC':2,'DIFF':2,'T1':2,'T2':2,'FLAIR':2,'ADC':2,'SWI':2}
#         device = "cpu"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
        labeled, labeled_images = defaultdict(list), defaultdict(list)
        modalities = ['DIFF','T2','T1', 'FLAIR', 'SWI']
        sub_modalities = ['T1','ADC']
   
        data_transforms = transforms.Compose(
            [resize_2Dimage(256), ZscoreNormalization(), ToTensor()])
        for m in modalities:
            model = load_checkpoint(checkpoints[m])
            test_dataset = MRClassifierDataset_test(
                images=for_inference, dummy=os.path.join(RESOURCES_PATH, 'random.nii.gz'),
                transform=data_transforms)  
            test_dataloader = DataLoader(
                test_dataset, batch_size=1, shuffle=False, num_workers=1)
             
            for _, data in enumerate(test_dataloader):
                 
                inputs = data['image']
                img_name = data['name']
                     
                inputs = inputs.to(device)
                output = model(inputs)
                prob = output.data.cpu().numpy()
                actRange = abs(prob[0][0])+abs(prob[0][1])
                index = output.data.cpu().numpy().argmax()
                     
                if index == 0 and actRange > th[m]:
                    labeled[img_name[0]].append([m,actRange])
         
        for key in labeled.keys():
            if len(labeled[key])>1:
                if labeled[key][0][1]>labeled[key][1][1]:
                    del labeled[key][1]
                else:
                    del labeled[key][0]
             
                         
        for key in labeled.keys():
            labeled_images[labeled[key][0][0]].append([key,labeled[key][0][1]])
                 
        labeled_subImages, labeled_s = defaultdict(list), defaultdict(list)
        #labeled_images = defaultdict(list)
        for m in sub_modalities:
            model = load_checkpoint(sub_checkpoints[m])
            if m =='ADC':
                list_images = [i[0] for i in labeled_images['DIFF']]
            else:
                list_images = [i[0] for i in labeled_images[m]]
            test_dataset = MRClassifierDataset_test(
                images=list_images, dummy=os.path.join(RESOURCES_PATH, 'random.nii.gz'),
                transform=data_transforms)  
            test_dataloader = DataLoader(
                test_dataset, batch_size = 1, shuffle=False, num_workers=8)
            for _, data in enumerate(test_dataloader):
                 
                inputs = data['image']
                img_name = data['name']
                     
                inputs = inputs.to(device)
                output = model(inputs)
                prob = output.data.cpu().numpy()
                actRange = abs(prob[0][0])+abs(prob[0][1])
                index = output.data.cpu().numpy().argmax()
                if m =='ADC':
                    if index == 0 and actRange > th[m]:
                        labeled_s[img_name[0]].append([m,actRange])
                    else:
                        labeled_s[img_name[0]].append([m+'KM',actRange])
                    continue
                             
                if index == 0:
                    labeled_s[img_name[0]].append([m,labeled[img_name[0]][0][1]])
                else:
                    labeled_s[img_name[0]].append([m+'KM',labeled[img_name[0]][0][1]])
                         
        for key in labeled_s.keys():
            if labeled_s[key][0][0] != 'ADCKM':
                labeled_subImages[labeled_s[key][0][0]].append([key,labeled_s[key][0][1]])
        for key in labeled_images.copy():
            #if key =='DIFF' or key=='T1' or key=='T2':
            if key == 'T1' or key == 'DIFF':
                del labeled_images[key]
   
        self.labelled_images = {**labeled_subImages, **labeled_images}
#         with open('/home/fsforazz/ww.pickle', 'wb') as f:
#             pickle.dump(self.labelled_images, f, protocol=pickle.HIGHEST_PROTOCOL)
          
#         with open('/home/fsforazz/ww.pickle', 'rb') as handle:
#             self.labelled_images = pickle.load(handle)

        to_remove = []
#         for key in self.labelled_images.keys():
#             labeled_list = [j[0][0:-7] for j in self.labelled_images[key]]
        for i in for_inference:
            image_dir = '/'.join(i.split('/')[:-1])
            to_remove = to_remove + [x for x in glob.glob(image_dir+'/*')
                                     if '.json' in x  or '.bval' in x
                                     or '.bvec' in x]
#             if i[0:-7] not in labeled_list:
#                 label_move_image(i[0:-7], 'Unclassifiable', image_dir)
        for f in to_remove:
            if os.path.isfile(f):
                os.remove(f)

#         self.unclassifiable = [x for x in for_inference if x[0:-7] not in labeled_list]
        for key in self.labelled_images.keys():
            for cm in self.labelled_images[key]:
                indices = [i for i, x in enumerate(cm[0]) if x == "/"]
                dirName = os.path.join(output_dir, cm[0][indices[-3]+1:indices[-1]], key)
                print(dirName)
                create_move_toDir(cm[0], dirName, cm[1])


        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.out_folder):
            outputs['out_folder'] = os.path.abspath(
                self.inputs.out_folder)

        return outputs

