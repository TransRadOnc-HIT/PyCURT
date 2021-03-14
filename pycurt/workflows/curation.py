import nipype
import os
from pycurt.interfaces.utils import DicomCheck, ConversionCheck, GetRefRTDose,\
     FileCheck, FolderMerge
from nipype.interfaces.dcm2nii import Dcm2niix
from pycurt.interfaces.plastimatch import DoseConverter
from core.workflows.base import BaseWorkflow
from pycurt.interfaces.utils import FolderPreparation, FolderSorting
from pycurt.interfaces.custom import RTDataSorting, ImageClassification
from nipype.interfaces.utility import Merge


POSSIBLE_SEQUENCES = ['t1', 'ct1', 't1km', 't2', 'flair', 'adc', 'swi', 'rtct',
                      'rtdose', 'rtplan', 'rtstruct']

PARENT_DIR =  '/media/fsforazz/Samsung_T5/mr_class/'

bp_cp = {'abd-pel':PARENT_DIR+'/checkpoints_new/bp-class/abd-pel_other_09402.pth',
                'lung':PARENT_DIR+'/checkpoints_new/bp-class/lung_other_09957.pth',
                'hnc':PARENT_DIR+'/checkpoints_new/bp-class/hnc_other_09974.pth',
                'wb':PARENT_DIR+'/checkpoints_new/bp-class/wb_other_09775.pth'
                   }

bp_sub_cp = {
                'wb':PARENT_DIR + '/checkpoints_new/bp-class/wb_wbh0_09213.pth',
                'hnc': PARENT_DIR + '/checkpoints_new/bp-class/brain_hnc_0.9425.pth',
}

mr_cp = {'ADC':PARENT_DIR+'/checkpoints_new/mr-abd/checkpoint_ADCvall_anal_11112019_0995_nodataaug.pth',
                'T1':PARENT_DIR+'/checkpoints_new/mr-abd/checkpoint_T1vall_anal_11112019_08788_nodataaug.pth',
                'T2':PARENT_DIR+'/checkpoints_new/mr-abd/checkpoint_T2vall_anal_11112019_08789_nodataaug.pth'
                   }

mr_sub_cp = {
                'T1':PARENT_DIR + '/checkpoints_new/mr-abd/checkpoint_T1vT1Km_anal_11112019_0882_nodataaug.pth',
}

class DataCuration(BaseWorkflow):
    
    @staticmethod
    def workflow_inputspecs():

        input_specs = {}
        input_specs['format'] = ''
        input_specs['dependencies'] = {}
        input_specs['suffix'] = ['']
        input_specs['prefix'] = []
        input_specs['data_formats'] = {'': ''}

        return input_specs

    @staticmethod
    def workflow_outputspecs():

        output_specs = {}
        output_specs['format'] = '.nii.gz'
        output_specs['suffix'] = ['']
        output_specs['prefix'] = []

        return output_specs

    def sorting_workflow(self, subject_name_position=-3, renaming=False,
                         mr_classiffication=True, mrclass_cp=mr_cp,
                         mrclass_sub_cp=mr_sub_cp, bp_class_cp=bp_cp,
                         bp_class_sub_cp=bp_sub_cp, bp='abd-pel',):

        if bp not in ['hnc', 'abd-pel'] and mr_classiffication:
            mr_classiffication = False

        nipype_cache = os.path.join(self.nipype_cache, 'data_sorting')
        result_dir = self.result_dir

        workflow = nipype.Workflow('sorting_workflow', base_dir=nipype_cache)
        datasink = nipype.Node(nipype.DataSink(base_directory=result_dir),
                               "datasink")

        file_check = nipype.Node(interface=FileCheck(), name='fc')
        file_check.inputs.input_dir = self.input_dir
        file_check.inputs.subject_name_position = subject_name_position
        file_check.inputs.renaming = renaming
        prep = nipype.MapNode(interface=FolderPreparation(), name='prep',
                              iterfield=['input_list'])
        sort = nipype.MapNode(interface=FolderSorting(), name='sort',
                              iterfield=['input_dir'])
        bp_class = nipype.MapNode(interface=ImageClassification(),
                                  name='bpclass',
                                  iterfield=['images2label'])
        bp_class.inputs.checkpoints = bp_class_cp
        bp_class.inputs.sub_checkpoints = bp_class_sub_cp
        bp_class.inputs.body_part = bp
        bp_class.inputs.network = 'bpclass'
        mr_rt_merge = nipype.MapNode(interface=Merge(2), name='mr_rt_merge',
                                    iterfield=['in1', 'in2'])
        mr_rt_merge.inputs.ravel_inputs = True
        merging = nipype.Node(interface=FolderMerge(), name='merge')
        if mr_classiffication:
            if mrclass_cp is None or mrclass_sub_cp is None:
                raise Exception('MRClass weights were not provided, MR image '
                                'classification cannot be performed!')
            mrclass = nipype.MapNode(interface=ImageClassification(), name='mrclass',
                                     iterfield=['images2label', 'out_folder'])
            mrclass.inputs.checkpoints = mrclass_cp
            mrclass.inputs.sub_checkpoints = mrclass_sub_cp
            mrclass.inputs.network = 'mrclass'
        else:
            mr_rt_merge.inputs.in1 = None
        rt_sorting = nipype.MapNode(interface=RTDataSorting(), name='rt_sorting',
                                    iterfield=['input_dir'])

        workflow.connect(file_check, 'out_list', prep, 'input_list')
        workflow.connect(prep, 'out_folder', sort, 'input_dir')
        workflow.connect(sort, 'out_folder', rt_sorting, 'input_dir')
        workflow.connect(sort, 'for_inference', bp_class, 'images2label')
        if mr_classiffication:
            workflow.connect(bp_class, 'labeled_images', mrclass, 'images2label')
            workflow.connect(bp_class, 'out_folder', mrclass, 'out_folder')
            workflow.connect(mrclass, 'out_folder', mr_rt_merge, 'in1')

            workflow.connect(rt_sorting, 'out_folder', mr_rt_merge, 'in2')
            workflow.connect(mr_rt_merge, 'out', merging, 'input_list')
            workflow.connect(merging, 'out_folder', datasink, '@rt_sorted')
        else:
            workflow.connect(rt_sorting, 'out_folder', datasink, '@rt_sorted')
            substitutions = [('_rt_sorting\d+/', '')]
            datasink.inputs.regexp_substitutions = substitutions
        
        return workflow

    def convertion_workflow(self):
        
        self.datasource()

        datasource = self.data_source
        dict_sequences = self.dict_sequences
        nipype_cache = self.nipype_cache
        result_dir = self.result_dir
        sub_id = self.sub_id

        toprocess = {**dict_sequences['MR-RT'], **dict_sequences['OT']}
        workflow = nipype.Workflow('data_convertion_workflow',
                                   base_dir=nipype_cache)
        datasink = nipype.Node(nipype.DataSink(base_directory=result_dir),
                               "datasink")
        substitutions = [('subid', sub_id)]
        substitutions += [('results/', '{}/'.format(self.workflow_name))]
        substitutions += [('checked_dicoms', 'RTSTRUCT_used')]
        datasink.inputs.substitutions = substitutions

        for key in toprocess:
            files = []
            if toprocess[key]['scans'] is not None:
                files = files + toprocess[key]['scans']
            for el in files:
                el = el.strip(self.extention)
                node_name = '{0}_{1}'.format(key, el)
                dc = nipype.Node(interface=DicomCheck(),
                                 name='{}_dc'.format(node_name))
                workflow.connect(datasource, node_name, dc, 'dicom_dir')
                converter = nipype.Node(interface=Dcm2niix(),
                                        name='{}_convert'.format(node_name))
                converter.inputs.compress = 'y'
                converter.inputs.philips_float = False
                if el == 'CT':
                    converter.inputs.merge_imgs = True
                else:
                    converter.inputs.merge_imgs = False
                check = nipype.Node(interface=ConversionCheck(),
                                    name='{}_cc'.format(node_name))
                workflow.connect(dc, 'outdir', converter, 'source_dir')
                workflow.connect(dc, 'scan_name', converter, 'out_filename')
                workflow.connect(dc, 'scan_name', check, 'file_name')
                workflow.connect(converter, 'converted_files', check, 'in_file')
                workflow.connect(check, 'out_file', datasink,
                     'results.subid.{0}.@{1}_converted'.format(key, el))

                check = nipype.Node(interface=ConversionCheck(),
                                    name='{}_cc'.format(node_name))

        for key in dict_sequences['RT']:
            doses = []
            if dict_sequences['RT'][key]['phy_dose'] is not None:
                doses.append('{}_phy_dose'.format(key))
            if dict_sequences['RT'][key]['rbe_dose'] is not None:
                doses.append('{}_rbe_dose'.format(key))
            for el in doses:
                el = el.strip(self.extention)
                node_name = el.strip(self.extention)
                converter = nipype.Node(interface=DoseConverter(),
                                        name='{}_dose_conv'.format(node_name))
                dc = nipype.Node(interface=DicomCheck(),
                                 name='{}_dc'.format(node_name))
                workflow.connect(datasource, node_name, dc, 'dicom_dir')
                workflow.connect(dc, 'dose_file', converter, 'input_dose')
                workflow.connect(dc, 'scan_name', converter, 'out_name')
                workflow.connect(converter, 'out_file', datasink,
                    'results.subid.{0}.@{1}_converted'.format(key, el))
            if dict_sequences['RT'][key]['ot_dose'] is not None:
                el = '{}_ot_dose'.format(key)
                node_name = el.strip(self.extention)
                converter = nipype.Node(interface=DoseConverter(),
                                        name='{}_convert'.format(node_name))
                get_dose = nipype.Node(interface=GetRefRTDose(),
                                       name='{}_get_dose'.format(node_name))
                workflow.connect(datasource, node_name, get_dose, 'doses')
                workflow.connect(get_dose, 'dose_file', converter, 'input_dose')
                converter.inputs.out_name = 'Unused_RTDOSE.nii.gz'
                workflow.connect(converter, 'out_file', datasink,
                    'results.subid.{0}.@{1}_converted'.format(key, el))
            if dict_sequences['RT'][key]['rtct'] is not None:
                el = '{}_rtct'.format(key)
                node_name = el.strip(self.extention)
                converter = nipype.Node(interface=Dcm2niix(),
                                        name='{}_convert'.format(node_name))
                converter.inputs.compress = 'y'
                converter.inputs.philips_float = False
                converter.inputs.merge_imgs = True
                dc = nipype.Node(interface=DicomCheck(),
                                 name='{}_dc'.format(node_name))
                workflow.connect(datasource, node_name, dc, 'dicom_dir')
                check = nipype.Node(interface=ConversionCheck(),
                                    name='{}_cc'.format(node_name))
                workflow.connect(dc, 'outdir', converter, 'source_dir')
                workflow.connect(dc, 'scan_name', converter, 'out_filename')
                workflow.connect(dc, 'scan_name', check, 'file_name')
                workflow.connect(converter, 'converted_files', check, 'in_file')
                workflow.connect(check, 'out_file', datasink,
                    'results.subid.{0}.@{1}_converted'.format(key, el))
            if dict_sequences['RT'][key]['rtstruct'] is not None:
                el = '{}_rtstruct'.format(key)
                node_name = el.strip(self.extention)
                dc = nipype.Node(interface=DicomCheck(),
                                 name='{}_dc'.format(node_name))
                workflow.connect(datasource, node_name, dc, 'dicom_dir')
                workflow.connect(dc, 'outdir', datasink,
                                 'results.subid.{0}.@rtstruct'.format(key, el))
    
        return workflow

    def workflow_setup(self, data_sorting=False, subject_name_position=-3,
                       renaming=False, mr_classiffication=True, mrclass_cp=mr_cp,
                       mrclass_sub_cp=mr_sub_cp):

        if data_sorting:
            workflow = self.sorting_workflow(
                subject_name_position=subject_name_position,
                renaming=renaming, mr_classiffication=mr_classiffication,
                mrclass_cp=mrclass_cp, mrclass_sub_cp=mrclass_sub_cp)

        else:
            workflow = self.convertion_workflow()

        return workflow
