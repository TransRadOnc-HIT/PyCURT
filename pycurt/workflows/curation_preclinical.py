import nipype
import os
from pycurt.interfaces.utils import FileCheck, FolderPreparation, PreclinicalSink
from pycurt.interfaces.custom import MouseCropping
from core.workflows.base import BaseWorkflow


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

    def sorting_workflow(self, subject_name_position=-3):

        result_dir = self.result_dir
        nipype_cache = os.path.join(self.nipype_cache, 'data_sorting')

        workflow = nipype.Workflow('sorting_workflow', base_dir=nipype_cache)

        file_check = nipype.Node(interface=FileCheck(), name='fc')
        file_check.inputs.input_dir = self.input_dir
        file_check.inputs.subject_name_position = subject_name_position
        file_check.inputs.renaming = False
        
        prep = nipype.MapNode(interface=FolderPreparation(), name='prep',
                              iterfield=['input_list'])
        
        crop = nipype.MapNode(interface=MouseCropping(), name='crop',
                              iterfield=['ct'])
        
        datasink = nipype.Node(PreclinicalSink(), name="datasink")
        datasink.inputs.out_folder = result_dir
        
        workflow.connect(crop, 'cropped_dir', datasink, 'tosink')
        workflow.connect(file_check, 'out_list', prep, 'input_list')
        workflow.connect(prep, 'to_crop', crop, 'ct')
        
        return workflow
    
    def workflow_setup(self, subject_name_position=-3):

        workflow = self.sorting_workflow(
            subject_name_position=subject_name_position)

        return workflow
