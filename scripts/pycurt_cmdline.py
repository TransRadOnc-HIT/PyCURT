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
                        if x in ['hnc', 'abd-pel']]
        bpclass_ct_cp, bpclass_ct_sub_cp = download_cl_network_weights(
            todownload='bpclass_ct')
        if mr_body_part:
            bpclass_mr_cp, bpclass_mr_sub_cp = download_cl_network_weights(
                todownload='bpclass_mr')
            mrclass_cp, mrclass_sub_cp = download_cl_network_weights(
                todownload='mrclass_{}'.format(mr_body_part[0]))
        else:
            bpclass_mr_cp, bpclass_mr_sub_cp = None, None
            mrclass_cp, mrclass_sub_cp = None, None

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
            bp_class_ct_cp=bpclass_ct_cp, bp_class_ct_sub_cp=bpclass_ct_sub_cp,
            bp_class_mr_cp=bpclass_mr_cp, bp_class_mr_sub_cp=bpclass_mr_sub_cp)
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
                if os.path.isdir(wd):
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
