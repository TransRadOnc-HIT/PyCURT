"Script to run PyCURT data sorting and/or curation from command line"
import os
import argparse
import yaml
from pycurt.workflows.curation import DataCuration
from pycurt.workflows.rt import RadioTherapy
from pycurt.utils.config import (
    create_subject_list, download_cl_network_weights,
    check_free_space)
from nipype import config
cfg = dict(execution={'hash_method': 'timestamp'})
config.update_config(cfg)

PARENT_DIR =  '/media/fsforazz/T7/classification_checkpoints'

bpclass_ct_cp = {'abd-pel':PARENT_DIR+'/bp_class_ct/checkpoint_abd-pel_acc_0.97.pth',
            'lung':PARENT_DIR+'/bp_class_ct/checkpoint_lung_acc_0.97.pth',
            'hnc':PARENT_DIR+'/bp_class_ct/checkpoint_hnc_acc_0.99.pth'}

bpclass_mr_cp = {'abd-pel':PARENT_DIR+'/bp_class_mr/checkpoint_mr_abd-pel_acc_0.94.pth',
            'hnc':PARENT_DIR+'/bp_class_mr/checkpoint_mr_hnc_acc_0.94.pth'}

mrclass_cp = {'T1': PARENT_DIR + '/mrclass/T1_other.pth',
               'T2': PARENT_DIR +'/mrclass/T2_other.pth',
               'FLAIR': PARENT_DIR+'/mrclass/FLAIR_other.pth',
               'SWI': PARENT_DIR+'/mrclass/SWI_other.pth',
               'ADC': PARENT_DIR+'/mrclass/ADC_other.pth',
               }
mrclass_sub_cp = { 'T1': PARENT_DIR  + '/mrclass/T1_T1KM.pth'}


def main():

    PARSER = argparse.ArgumentParser()
    
    PARSER.add_argument('--input-dir', '-i', type=str, required=True,
                        help=('Exisisting directory with the subject(s) to process'))
    PARSER.add_argument('--work-dir', '-w', type=str, required=True,
                        help=('Directory where to store the results.'))
    PARSER.add_argument('--config-file', '-c', type=str, required=True,
                        help=('PyCURT configuration file in YAML format.'))

    ARGS = PARSER.parse_args()
    
    with open(ARGS.config_file) as f: 
        PARAMETER_CONFIG = yaml.safe_load(f)

    BASE_DIR = ARGS.input_dir

    sub_list, BASE_DIR = create_subject_list(BASE_DIR, subjects_to_process=[])

    if PARAMETER_CONFIG['data_sorting']:
        mr_body_part = [x for x in PARAMETER_CONFIG['body_part']
                        if x in ['hnc']]
#         bpclass_ct_cp, bpclass_ct_sub_cp = download_cl_network_weights(
#             todownload='bpclass_ct')
#         if mr_body_part:
#             bpclass_mr_cp, bpclass_mr_sub_cp = download_cl_network_weights(
#                 todownload='bpclass_mr')
#             mrclass_cp, mrclass_sub_cp = download_cl_network_weights(
#                 todownload='mrclass_{}'.format(mr_body_part[0]))
#         else:
#             bpclass_mr_cp, bpclass_mr_sub_cp = None, None
#             mrclass_cp, mrclass_sub_cp = None, None

        if not os.path.isdir(os.path.join(ARGS.work_dir, 'nipype_cache')):
            check_free_space(BASE_DIR, ARGS.work_dir)
        workflow = DataCuration(
            sub_id='', input_dir=BASE_DIR, work_dir=ARGS.work_dir,
            process_rt=True, cores=PARAMETER_CONFIG['nummber-of-cores'], local_sink=False)
        wf = workflow.workflow_setup(
            data_sorting=True,
            subject_name_position=PARAMETER_CONFIG['subject-name-position'],
            renaming=PARAMETER_CONFIG['renaming'],
            mrrt_max_time_diff=PARAMETER_CONFIG['mrrt-max-time-diff'],
            rert_max_time=PARAMETER_CONFIG['replanning_rt-max-time-diff'],
            body_parts=PARAMETER_CONFIG['body_part'],
            mrclass_cp=mrclass_cp, mrclass_sub_cp=mrclass_sub_cp,
            bp_class_ct_cp=bpclass_ct_cp, bp_class_mr_cp=bpclass_mr_cp,
            bp_class_ct_th=PARAMETER_CONFIG['bp_class_ct_th'],
            bp_class_mr_th=PARAMETER_CONFIG['bp_class_mr_th'],
            mr_classification=PARAMETER_CONFIG['mr_classification'])
        workflow.runner(wf)
        BASE_DIR = os.path.join(ARGS.work_dir, 'workflows_output', 'Sorted_Data')
        sub_list, BASE_DIR = create_subject_list(BASE_DIR, subjects_to_process=[])

    if PARAMETER_CONFIG['data_curation']:
        for sub_id in sub_list:
            print('Processing subject {}'.format(sub_id))
    
            workflow = DataCuration(
                sub_id=sub_id, input_dir=BASE_DIR, work_dir=ARGS.work_dir,
                process_rt=True, local_basedir=PARAMETER_CONFIG['local-basedir'],
                local_project_id=PARAMETER_CONFIG['local-project-id'],
                local_sink=PARAMETER_CONFIG['local-sink'],
                cores=PARAMETER_CONFIG['nummber-of-cores'])
            wf = workflow.workflow_setup()
            if wf.list_node_names():
                workflow.runner(wf)
            if PARAMETER_CONFIG['extract-rts']:
                wd = os.path.join(ARGS.work_dir, 'workflows_output', 'DataCuration')
                if os.path.isdir(os.path.join(wd, sub_id)):
                    workflow = RadioTherapy(
                        sub_id=sub_id, input_dir=wd, work_dir=ARGS.work_dir,
                        process_rt=True, roi_selection=PARAMETER_CONFIG['select-rts'],
                        local_basedir=PARAMETER_CONFIG['local-basedir'],
                        local_project_id=PARAMETER_CONFIG['local-project-id'],
                        local_sink=PARAMETER_CONFIG['local-sink'],
                        cores=PARAMETER_CONFIG['nummber-of-cores'])
                    wf = workflow.workflow_setup()
                    if wf.list_node_names():
                        workflow.runner(wf)

    print('Done!')


if __name__ == "__main__":
    main()
