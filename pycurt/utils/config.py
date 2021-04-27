import os
import glob
from pycurt.utils.utils import get_files, untar
import subprocess
import shutil
import time


def check_free_space(in_path, out_path):

    print('Start checking the size of the input folder to see if there is enough '
          'free space to run PyCURT data sorting. This might take a while, '
          'depending on the input data size')
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
              'the process might fail. Consider changing the working directory or '
              ' free some additional space'.format(in_size, in_size*2, out_size))
        print('The sorting will start in 20 seconds...')
        time.sleep(20)


def download_cl_network_weights(
        weights_dir=None, todownload='mrclass_hnc',
        url=('http://www.oncoexpress.de/software/pycurt'
            '/classification_networks_weights/'),
        ):

    if weights_dir is None:
        home = os.path.expanduser("~")
        weights_dir = os.path.join(home, '.weights_dir')

    try:
        TAR_FILE = get_files(os.path.join(url, todownload+'.tar.gz'),
                             weights_dir, todownload)
        untar(TAR_FILE)
    except:
        raise Exception('Unable to download: {}!'.format(todownload))

    weights = [w for w in sorted(glob.glob(os.path.join(weights_dir, todownload, '*.pth')))]
    
    checkpoints = {}
    sub_checkpoints = {}

    if todownload == 'mrclass_hnc':
        checkpoints['T1'] = [x for x in weights if 'T1_other' in x][0]
        checkpoints['T2'] = [x for x in weights if 'T2_other' in x][0]
        checkpoints['FLAIR'] = [x for x in weights if 'FLAIR_other' in x][0]
        checkpoints['ADC'] = [x for x in weights if 'ADC_other' in x][0]
        checkpoints['SWI'] = [x for x in weights if 'SWI_other' in x][0]
        
        sub_checkpoints['T1'] = [x for x in weights if 'T1_T1KM' in x][0]
    elif todownload == 'mrclass_abd-pel':
        checkpoints['T1'] = [x for x in weights if 'T1_other' in x][0]
        checkpoints['T2'] = [x for x in weights if 'T2_other' in x][0]
        checkpoints['ADC'] = [x for x in weights if 'ADC_other' in x][0]
        
        sub_checkpoints['T1'] = [x for x in weights if 'T1_T1KM' in x][0]
    elif todownload == 'bpclass_ct':
        checkpoints['abd-pel'] = [x for x in weights if 'abd-pel_other' in x][0]
        checkpoints['lung'] = [x for x in weights if 'lung_other' in x][0]
        checkpoints['hnc'] = [x for x in weights if 'hnc_other' in x][0]
        checkpoints['wb'] = [x for x in weights if 'wb_other' in x][0]
        
        sub_checkpoints['wb'] = [x for x in weights if 'wb_wbhead' in x][0]
        sub_checkpoints['hnc'] = [x for x in weights if 'brain_hnc' in x][0]
    elif todownload == 'bpclass_mr':
        checkpoints['abd-pel'] = [x for x in weights if 'abd-pel_other' in x][0]
        checkpoints['hnc'] = [x for x in weights if 'hnc_other' in x][0]

        sub_checkpoints['hnc'] = [x for x in weights if 'brain_hnc' in x][0]
    else:
        print('Requested network weights are not available')
        
    
    return checkpoints, sub_checkpoints


def create_subject_list(base_dir, subjects_to_process=[]):
    
    if os.path.isdir(base_dir):
        sub_list = [x for x in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, x))]
        if subjects_to_process:
            sub_list = [x for x in sub_list if x in subjects_to_process]

    return sub_list, base_dir
