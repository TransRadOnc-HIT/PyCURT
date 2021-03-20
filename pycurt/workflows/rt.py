"File containing all the workflows related to RadioTherapy"
import nipype
from pycurt.interfaces.plastimatch import RTStructureCoverter
from pycurt.interfaces.utils import CheckRTStructures, MHA2NIIConverter
from core.workflows.base import BaseWorkflow


POSSIBLE_SEQUENCES = ['t1', 'ct1', 't1km', 't2', 'flair', 'adc', 'swi', 'rtct',
                      'rtdose', 'rtplan', 'rtstruct']


class RadioTherapy(BaseWorkflow):

    def __init__(self, regex=None, roi_selection=False, **kwargs):
        
        super().__init__(**kwargs)
        self.regex = regex
        self.roi_selection = roi_selection
    
    @staticmethod
    def workflow_inputspecs(additional_inputs=None):

        input_specs = {}
        input_specs['format'] = '.nii.gz'
        input_specs['inputs'] = {
            '': {'mandatory': True, 'format': '.nii.gz', 'dependency': None,
                 'possible_sequences': POSSIBLE_SEQUENCES, 'multiplicity': 'rt',
                 'composite': None}}
        input_specs['dependencies'] = {}
        input_specs['suffix'] = ['']
        input_specs['prefix'] = []
        input_specs['data_formats'] = {'': '.nii.gz'}
        input_specs['additional_inputs'] = additional_inputs

        return input_specs

    @staticmethod
    def workflow_outputspecs():

        output_specs = {}
        dict_outputs = {'_preproc': {
            'possible_sequences': POSSIBLE_SEQUENCES, 'format': '.nii.gz',
            'multiplicity': 'all', 'composite': None}}
        output_specs['outputs'] = dict_outputs

    def workflow(self):

#         self.datasource()
        datasource = self.data_source
        dict_sequences = self.dict_sequences
        nipype_cache = self.nipype_cache
        result_dir = self.result_dir
        sub_id = self.sub_id
        regex = self.regex
        roi_selection = self.roi_selection
        
        workflow = nipype.Workflow('rtstruct_extraction_workflow',
                                   base_dir=nipype_cache)
        datasink = nipype.Node(nipype.DataSink(base_directory=result_dir),
                               "datasink")
        substitutions = [('subid', sub_id)]
        substitutions += [('results/', '{}/'.format(self.workflow_name))]
        substitutions += [('_mha_convert/', '/')]

        rt_sessions = dict_sequences['RT']
        for key in rt_sessions:
            rt_files = rt_sessions[key]
            if rt_files['phy_dose'] is not None:
                dose_name = '{0}_phy_dose'.format(key)
            elif rt_files['rbe_dose'] is not None:
                dose_name = '{0}_rbe_dose'.format(key)
            elif rt_files['ot_dose'] is not None:
                dose_name = '{0}_ot_dose'.format(key)
            else:
                roi_selection = False
            
            ss_convert = nipype.Node(interface=RTStructureCoverter(),
                                     name='ss_convert')
            mha_convert = nipype.Node(interface=MHA2NIIConverter(),
                                      name='mha_convert')

            if roi_selection:
                select = nipype.Node(interface=CheckRTStructures(),
                                     name='select_gtv')
                workflow.connect(mha_convert, 'out_files', select, 'rois')
                workflow.connect(datasource, dose_name, select, 'dose_file')
                workflow.connect(select, 'checked_roi', datasink,
                                 'results.subid.{}.@masks'.format(key))
            else:
                workflow.connect(mha_convert, 'out_files', datasink,
                                 'results.subid.{}.@masks'.format(key))

            datasink.inputs.substitutions =substitutions
        
            workflow.connect(datasource, '{0}_rtct'.format(key), ss_convert, 'reference_ct')
            workflow.connect(datasource, '{0}_rtstruct'.format(key), ss_convert, 'input_ss')
            workflow.connect(ss_convert, 'out_structures', mha_convert, 'input_folder')
            
#         if datasource is not None:
# 
#             workflow = nipype.Workflow('rtstruct_extraction_workflow', base_dir=nipype_cache)
#         
#             datasink = nipype.Node(nipype.DataSink(base_directory=result_dir), "datasink")
#             substitutions = [('subid', sub_id)]
#             substitutions += [('results/', '{}/'.format(self.workflow_name))]
#     
#             ss_convert = nipype.MapNode(interface=RTStructureCoverter(),
#                                        iterfield=['reference_ct', 'input_ss'],
#                                        name='ss_convert')
#             mha_convert = nipype.MapNode(interface=MHA2NIIConverter(),
#                                          iterfield=['input_folder'],
#                                          name='mha_convert')
#             
#             if roi_selection:
#                 select = nipype.MapNode(interface=CheckRTStructures(),
#                                         iterfield=['rois', 'dose_file'],
#                                         name='select_gtv')
#                 workflow.connect(mha_convert, 'out_files', select, 'rois')
#                 workflow.connect(datasource, 'rt_dose', select, 'dose_file')
#                 workflow.connect(select, 'checked_roi', datasink,
#                                  'results.subid.@masks')
#             else:
#                 workflow.connect(mha_convert, 'out_files', datasink,
#                                  'results.subid.@masks')
# 
#             for i, session in enumerate(self.rt['session']):
#                 substitutions += [(('_select_gtv{}/'.format(i), session+'/'))]
#                 substitutions += [(('_voxelizer{}/'.format(i), session+'/'))]
#                 substitutions += [(('_mha_convert{}/'.format(i), session+'/'))]
# 
#             datasink.inputs.substitutions =substitutions
#         
#             workflow.connect(datasource, 'rtct_nifti', ss_convert, 'reference_ct')
#             workflow.connect(datasource, 'rts_dcm', ss_convert, 'input_ss')
#             workflow.connect(ss_convert, 'out_structures', mha_convert, 'input_folder')
#     
#             workflow = self.datasink(workflow, datasink)
#         else:
#             workflow = nipype.Workflow('rtstruct_extraction_workflow', base_dir=nipype_cache)

        return workflow
