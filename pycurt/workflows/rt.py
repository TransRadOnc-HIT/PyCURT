"File containing all the workflows related to RadioTherapy"
import nipype
from pycurt.interfaces.plastimatch import RTStructureCoverter
from pycurt.interfaces.utils import CheckRTStructures, MHA2NIIConverter
from core.workflows.base import BaseWorkflow


class RadioTherapy(BaseWorkflow):

    def __init__(self, regex=None, roi_selection=False, **kwargs):
        
        super().__init__(**kwargs)
        self.regex = regex
        self.roi_selection = roi_selection

    def datasource(self, **kwargs):

        self.database()
        rt = self.rt
        rt_dose = None
        if self.rt is not None:
            if rt:
                if rt['physical']:
                    rt_dose = '*PHY*.nii.gz'
                elif rt['rbe']:
                    rt_dose = '*RBE*.nii.gz'
                elif rt['doses']:
                    rt_dose = 'Unused_RTDOSE.nii.gz'
                else:
                    if self.roi_selection:
                        self.roi_selection = False
            else:
                if self.roi_selection:
                    self.roi_selection = False
            
            field_template = dict()
            template_args = dict()
            if rt_dose is not None:
                field_template['rt_dose'] = '%s/%s/{}'.format(rt_dose)
                template_args['rt_dose'] = [['sub_id', 'rt']]
    
            field_template['rtct_nifti'] = '%s/%s/RTCT.nii.gz'
            template_args['rtct_nifti'] = [['sub_id', 'rt']]
            
            field_template['rts_dcm'] = '%s/%s/RTSTRUCT_used/*dcm'
            template_args['rts_dcm'] = [['sub_id', 'rt']]
    
            field_template['rois'] = '%s/%s/out_struct*'
            template_args['rois'] = [['sub_id', 'rt']]
    
            
            field_template.update(self.field_template)
            template_args.update(self.template_args)
            self.outfields = [x for x in field_template.keys()]
            self.field_template = field_template
            self.template_args = template_args
    
            self.data_source = self.create_datasource()
        else:
            self.data_source = None

    def workflow(self):

        self.datasource()
        datasource = self.data_source
        nipype_cache = self.nipype_cache
        result_dir = self.result_dir
        sub_id = self.sub_id
        regex = self.regex
        roi_selection = self.roi_selection
        if datasource is not None:

            workflow = nipype.Workflow('rtstruct_extraction_workflow', base_dir=nipype_cache)
        
            datasink = nipype.Node(nipype.DataSink(base_directory=result_dir), "datasink")
            substitutions = [('subid', sub_id)]
            substitutions += [('results/', '{}/'.format(self.workflow_name))]
    
            ss_convert = nipype.MapNode(interface=RTStructureCoverter(),
                                       iterfield=['reference_ct', 'input_ss'],
                                       name='ss_convert')
            mha_convert = nipype.MapNode(interface=MHA2NIIConverter(),
                                         iterfield=['input_folder'],
                                         name='mha_convert')
            
            if roi_selection:
                select = nipype.MapNode(interface=CheckRTStructures(),
                                        iterfield=['rois', 'dose_file'],
                                        name='select_gtv')
                workflow.connect(mha_convert, 'out_files', select, 'rois')
                workflow.connect(datasource, 'rt_dose', select, 'dose_file')
                workflow.connect(select, 'checked_roi', datasink,
                                 'results.subid.@masks')
            else:
                workflow.connect(mha_convert, 'out_files', datasink,
                                 'results.subid.@masks')

            for i, session in enumerate(self.rt['session']):
                substitutions += [(('_select_gtv{}/'.format(i), session+'/'))]
                substitutions += [(('_voxelizer{}/'.format(i), session+'/'))]
                substitutions += [(('_mha_convert{}/'.format(i), session+'/'))]

            datasink.inputs.substitutions =substitutions
        
            workflow.connect(datasource, 'rtct_nifti', ss_convert, 'reference_ct')
            workflow.connect(datasource, 'rts_dcm', ss_convert, 'input_ss')
            workflow.connect(ss_convert, 'out_structures', mha_convert, 'input_folder')
    
            workflow = self.datasink(workflow, datasink)
        else:
            workflow = nipype.Workflow('rtstruct_extraction_workflow', base_dir=nipype_cache)

        return workflow
