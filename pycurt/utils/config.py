import os
import glob
from pycurt.utils.utils import get_files, untar


def download_mrclass_weights(weights_dir=None,
                             url=('http://www.oncoexpress.de/software/pycurt/network_weights/mrclass/mrclass_weights.tar.gz')):

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
    checkpoints['DIFF'] = [x for x in weights if 'ADCvsAll' in x][0]
    checkpoints['SWI'] = [x for x in weights if 'SWIvsAll' in x][0]
    
    sub_checkpoints['T1'] = [x for x in weights if 'T1vsT1KM' in x][0]
    sub_checkpoints['ADC'] = [x for x in weights if 'ADCvsDiff' in x][0]
    
    return checkpoints, sub_checkpoints


def create_subject_list(base_dir, subjects_to_process=[]):
    
    if os.path.isdir(base_dir):
        sub_list = [x for x in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, x))]
        if subjects_to_process:
            sub_list = [x for x in sub_list if x in subjects_to_process]

    return sub_list, base_dir
