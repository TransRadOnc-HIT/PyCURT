import os
import glob
from pycurt.utils.utils import get_files, untar
import subprocess
import shutil
import time


def check_free_space(in_path, out_path):

    in_size = int(subprocess.check_output(['du','-sk', in_path])
                  .split()[0].decode('utf-8'))
    in_size = in_size*0.000977
    print("The size of the input directory is: {} Mb".format(in_size))
    par_dir = '/'.join(out_path.split('/')[:-2])
    _, _, out_size = shutil.disk_usage(par_dir)
    out_size = out_size*(9.537*(10**(-7)))
    print("The free space in the parent directory of the specified "
          "working directory is: {} Mb".format(out_size))

    if in_size*2 > out_size:
        print('WARNING! It seems there is not enough space to run the data sorting.'
              'In order to sort the provided data (size={0} Mb), PyCURT can need up to '
              '{1} Mb of free space, while the specified working directory seems to have '
              'only {2} Mb of available space. This is a rough estimation and at the end '
              'the space might be enough, but be warned that '
              'the process might fail.'.format(in_size, in_size*2, out_size))
        print('The sorting will start in 20 seconds...')
        time.sleep(20)


def download_mrclass_weights(
        weights_dir=None, url=(
            'http://www.oncoexpress.de/software/pycurt'
            '/network_weights/mrclass/mrclass_weights.tar.gz')):

    if weights_dir is None:
        home = os.path.expanduser("~")
        weights_dir = os.path.join(home, '.weights_dir')

    try:
        TAR_FILE = get_files(url, weights_dir, 'mrclass_weights')
        untar(TAR_FILE)
    except:
        raise Exception('Unable to download mrclass weights!')

    weights = [w for w in sorted(glob.glob(os.path.join(weights_dir, '*.pth')))]
    
    checkpoints = {}
    sub_checkpoints = {}
    
    checkpoints['T1'] = [x for x in weights if 'T1vsAll' in x][0]
    checkpoints['T2'] = [x for x in weights if 'T2vsAll' in x][0]
    checkpoints['FLAIR'] = [x for x in weights if 'FLAIRvsAll' in x][0]
    checkpoints['ADC'] = [x for x in weights if 'ADCvsAll' in x][0]
    checkpoints['SWI'] = [x for x in weights if 'SWIvsAll' in x][0]
    
    sub_checkpoints['T1'] = [x for x in weights if 'T1vsT1KM' in x][0]
    
    return checkpoints, sub_checkpoints


def create_subject_list(base_dir, subjects_to_process=[]):
    
    if os.path.isdir(base_dir):
        sub_list = [x for x in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, x))]
        if subjects_to_process:
            sub_list = [x for x in sub_list if x in subjects_to_process]

    return sub_list, base_dir


def parameters_config():
    
    parameters = {}
    
    """
    The body part(s) you are interested in. You can choose among:
    -hnc: head and neck
    -hncKM: head only (no neck)
    -abd-pel: abdominal-pelvic
    -wb: whole body (no head)
    -wbKM: whole body plus head
    -lung: lungs
    Multiple body parts of interest are allowed.
    Please be aware that the MR image classification works only 
    for hnc, hncKM and abd-pel. For the other parts, you will only
    get CT and RT data sorted.
    """
    parameters['body_part'] = ['hnc', 'hncKM']

    """
    If you do not want to run the data convertion after sorting,
    then set this to False.
    """
    parameters['data_curation'] = True

    """
    Change to True if you want to use the information stored
    in the DICOM header to rename the subject and sessions
    folders. Be warned that sometimes DICOMs do not have this
    information causing the software to ignore that file.
   """
    parameters['renaming'] = False

    """
    This is the maximum time, in days, between the radiotherapy
    session and the MR planning session. If not MR sessions are
    found BEFORE or on the same day of the RT, then PyCURT will
    check this number of days AFTER RT date to see if there are any
    MR session(s). It will take the first MR session in this time
    window, if any.
   """
    parameters['mrrt-max-time-diff'] = 15

    """
    PyCURT will treat any RT timepoint found 42 days after the firt 
    RT as replanning-RT. Any RT session after 42 days will be considered 
    as recurrent irradiation and the subject will be splitted.
    You can change this number based on your entity.
    """
    parameters['replanning_rt-max-time-diff'] = 42

    """
    Change this to False if you do not want to extract all the structures
    in the RT strucuture set (if any).
    """
    parameters['extract-rts'] = True

    """
    PyCURT will save ONLY the structure (within the RT structure set)
    that has the highest overlap with the dose distribution (if present).
    If you want to save ALL of them, change this to False.
    """
    parameters['select-rts'] = True

    """
    Change this to False if you DO NOT want to have a local database with
    all the outputs from the different workflows saved together. If False,
    each workflow will have its own output folder.
    """
    parameters['local-sink'] = True

    """
    If you create the local database, you can change the name of the
    project folder here.
    """
    parameters['local-project-id'] = 'PyCURT_sorting_database'

    """
    By default, the folder where the local database will be created is 
    the working directory. If you want to save it in a different folder
    you can provide the full path here.
    """
    parameters['local-basedir'] = ''

    return parameters
