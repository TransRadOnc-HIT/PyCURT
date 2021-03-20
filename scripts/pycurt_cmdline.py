"Script to run PyCURT data sorting and/or curation from command line"
import os
import argparse
from pycurt.workflows.curation import DataCuration
from pycurt.workflows.rt import RadioTherapy
from pycurt.utils.config import (
    create_subject_list, download_mrclass_weights,
    parameters_config, check_free_space)


def main():

    PARSER = argparse.ArgumentParser()
    
    PARSER.add_argument('--input_dir', '-i', type=str,
                        help=('Exisisting directory with the subject(s) to process'))
    PARSER.add_argument('--work_dir', '-w', type=str,
                        help=('Directory where to store the results.'))
    PARSER.add_argument('--num-cores', '-nc', type=int, default=0,
                        help=('Number of cores to use to run the registration workflow '
                              'in parallel. Default is 0, which means the workflow '
                              'will run linearly.'))
    PARSER.add_argument('--data_sorting', '-ds', action='store_true',
                        help=('Whether or not to sort the data before convertion. '
                              'If not, the software assumes you ran the folder sorting '
                              'before using PyCURT. Default is False'))
    PARSER.add_argument('--subject-name-position', '-np', type=int, default=-3,
                        help=('The position of the subject ID '
                              'in the image path has to be specified (assuming it will'
                              ' be the same for all the files). For example, '
                              'the position in the  subject ID (sub1) for a file called '
                              '"/mnt/sdb/tosort/sub1/session1/image.dcm", will be 4 '
                              '(or -3, remember that in Python'
                              ' numbering starts from 0). By default, is the third'
                              ' position starting from the end of the path.'))
    ARGS = PARSER.parse_args()
    
    PARAMETER_CONFIG = parameters_config()

    BASE_DIR = ARGS.input_dir

    sub_list, BASE_DIR = create_subject_list(BASE_DIR, subjects_to_process=[])

    if ARGS.data_sorting:
        checkpoints, sub_checkpoints = download_mrclass_weights()
        check_free_space(BASE_DIR, ARGS.work_dir)
        workflow = DataCuration(
            sub_id='', input_dir=BASE_DIR, work_dir=ARGS.work_dir,
            process_rt=True, cores=ARGS.num_cores, local_sink=False)
        wf = workflow.workflow_setup(
            data_sorting=True, subject_name_position=ARGS.subject_name_position,
            renaming=PARAMETER_CONFIG['renaming'],
            mrrt_max_time_diff=PARAMETER_CONFIG['mrrt-max-time-diff'],
            rert_max_time=PARAMETER_CONFIG['replanning_rt-max-time-diff'],
            body_parts=PARAMETER_CONFIG['body_part'])
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
                cores=ARGS.num_cores)
            wf = workflow.workflow_setup()
            if wf.list_node_names():
                workflow.runner(wf)
            if PARAMETER_CONFIG['extract-rts']:
                wd = os.path.join(ARGS.work_dir, 'workflows_output', 'DataCuration')
                workflow = RadioTherapy(
                    sub_id=sub_id, input_dir=wd, work_dir=ARGS.work_dir,
                    process_rt=True, roi_selection=PARAMETER_CONFIG['select-rts'],
                    local_basedir=PARAMETER_CONFIG['local-basedir'],
                    local_project_id=PARAMETER_CONFIG['local-project-id'],
                    local_sink=PARAMETER_CONFIG['local-sink'],
                    cores=ARGS.num_cores)
                wf = workflow.workflow_setup()
                if wf.list_node_names():
                    workflow.runner(wf)

    print('Done!')


if __name__ == "__main__":
    main()
