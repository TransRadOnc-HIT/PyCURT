"Script to run PyCURT data sorting for mouse clinical CT images from command line"
import argparse
from pycurt.workflows.curation_preclinical import DataCuration
from nipype import config
cfg = dict(execution={'hash_method': 'timestamp'})
config.update_config(cfg)


def main():

    PARSER = argparse.ArgumentParser()
    
    PARSER.add_argument('--input-dir', '-i', type=str, required=True,
                        help=('Exisisting directory with the subject(s) to process'))
    PARSER.add_argument('--work-dir', '-w', type=str, required=True,
                        help=('Directory where to store the results.'))
    PARSER.add_argument('--nummber-of-cores', '-c', type=int, default=0,
                        help=('Number of cores to use to run the registration workflow'
                              'in parallel. Default is 0, which means the workflow will'
                              ' run linearly.'))
    PARSER.add_argument('--subject-name-position', '-s', type=int, required=True,
                        help=('The position of the subject ID in the image path has '
                              'to be specified (assuming it will be the same for all '
                              'the files). For example, the position in the  '
                              'subject ID (sub1) for a file called'
                              ' "/mnt/sdb/tosort/sub1/session1/image.dcm", will be 4'
                              '(or -3, remember that in Python numbering starts from 0).'
                              ' By default, is the third position starting from the end'
                              ' of the path.'))

    ARGS = PARSER.parse_args()

    BASE_DIR = ARGS.input_dir

    workflow = DataCuration(
        sub_id='', input_dir=BASE_DIR, work_dir=ARGS.work_dir,
        process_rt=True, cores=ARGS.nummber_of_cores,
        local_sink=False)
    wf = workflow.workflow_setup(
        subject_name_position=ARGS.subject_name_position)
    workflow.runner(wf)

    print('Done!')


if __name__ == "__main__":
    main()
