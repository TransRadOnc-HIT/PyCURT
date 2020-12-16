#!/usr/bin/env python
"Script to run the PyCURT GUI"
import os
from pycurt.workflows.curation import DataCuration
from pycurt.utils.config import create_subject_list, download_mrclass_weights
from pycurt.utils.utils import create_pycurt_gui
from pycurt.workflows.rt import RadioTherapy


def main():
    values = create_pycurt_gui()
    
    BASE_DIR = values['in_path']
    
    sub_list, BASE_DIR = create_subject_list(BASE_DIR)
    
    if values['data_sorting']:
        if values['mrclass']:
            checkpoints, sub_checkpoints = download_mrclass_weights()
        else:
            checkpoints = None
            sub_checkpoints = None
    
        workflow = DataCuration(
            sub_id='', input_dir=BASE_DIR, work_dir=values['work_dir'],
            process_rt=True, cores=int(values['cores']))
        wf = workflow.workflow_setup(
            data_sorting=True, subject_name_position=int(values['sn_pos']),
            renaming=values['renaming'], mr_classiffication=values['mrclass'],
            checkpoints=checkpoints, sub_checkpoints=sub_checkpoints)
        workflow.runner(wf)
        BASE_DIR = os.path.join(values['work_dir'], 'workflows_output', 'Sorted_Data')
        sub_list = os.listdir(BASE_DIR)
    
    if values['data_curation']:
        for sub_id in sub_list:
        
            print('Processing subject {}'.format(sub_id))
        
            workflow = DataCuration(
                sub_id=sub_id, input_dir=BASE_DIR, work_dir=values['work_dir'],
                process_rt=True, local_basedir=values['db_path'],
                local_project_id=values['db_pid'], local_sink=values['local_db'],
                cores=int(values['cores']))
            wf = workflow.workflow_setup()
            if wf.list_node_names():
                workflow.runner(wf)
            if values['extract_rts']:
                wd = os.path.join(values['work_dir'], 'workflows_output', 'DataCuration')
                if os.path.isdir(os.path.join(wd, sub_id)):
                    workflow = RadioTherapy(
                        sub_id=sub_id, input_dir=wd, work_dir=values['work_dir'],
                        process_rt=True, roi_selection=values['select_roi'],
                        local_basedir=values['db_path'], cores=int(values['cores']),
                        local_project_id=values['db_pid'], local_sink=values['local_db'])
                    wf = workflow.workflow_setup()
                    if wf.list_node_names():
                        workflow.runner(wf)
    
    print('Done!')


if __name__ == "__main__":
    main()
    