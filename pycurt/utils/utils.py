import os
import pydicom
import glob
import re
import pydicom as pd
import requests
import tarfile
import PySimpleGUI as sg
import sys
import pickle


def check_dcm_dose(dcms):

    right_dcm = []
    for dcm in dcms:
        hd = pydicom.read_file(dcm)
        try:
            hd.GridFrameOffsetVector
            hd.pixel_array
            right_dcm.append(dcm)
        except:
            continue
    return right_dcm


def check_rtstruct(basedir, regex):

    data = []
    no_rt = []
    no_match = []
    for root, _, files in os.walk(basedir):
        for name in files:
            if (('RTDOSE' in name and name.endswith('.nii.gz'))
                    and os.path.isdir(os.path.join(root, 'RTSTRUCT_used'))):
                sub_name = root.split('/')[-2]
                try:
                    rts = glob.glob(os.path.join(root, 'RTSTRUCT_used', '*.dcm'))[0]
                    matching = check_rts(rts, regex)
                    if matching:
                        data.append(sub_name)
                    else:
                        no_match.append(sub_name)
                except IndexError:
                    print('No RTSTRUCT for {}'.format(root.split('/')[-2]))
                    no_rt.append(sub_name)
    return list(set(data))


def check_rts(rts, regex):

    ds = pd.read_file(rts)
    reg_expression = re.compile(regex)
    matching_regex = False
    for i in range(len(ds.StructureSetROISequence)):
        match = reg_expression.match(ds.StructureSetROISequence[i].ROIName)
        if match is not None:
            matching_regex = True
            break
    return matching_regex


def get_files(url, location, file, ext='.tar.gz'):

    if not os.path.isfile(os.path.join(location, file+ext)):
        if not os.path.isdir(os.path.join(location)):
            os.makedirs(os.path.join(location))
        r = requests.get(url)
        with open(os.path.join(location, file+ext), 'wb') as f:
            f.write(r.content)
        print(r.status_code)
        print(r.headers['content-type'])
        print(r.encoding)

    return os.path.join(location, file+ext)


def untar(fname):

    untar_dir = os.path.split(fname)[0]
    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname)
        tar.extractall(path=untar_dir)
        tar.close()
        print("Extracted in Current Directory")
    else:
        print("Not a tar.gz file: {}".format(fname))


def create_pycurt_gui():
    
    data_sorting = False
    sg.ChangeLookAndFeel('GreenTan')
    
    input_layout = [
        [sg.Text('Input folder', size=(20, 1), auto_size_text=False, justification='right', font=("Courier 10 Pitch", 20)),
         sg.InputText('', key='in_path', tooltip=(
             'Folder containing DICOM files in any structure.'), font=("Courier 10 Pitch", 20)),
         sg.FolderBrowse(font=("Courier 10 Pitch", 20))],
        [sg.Text('Working directory', size=(20, 1), auto_size_text=False,
                 justification='right', font=("Courier 10 Pitch", 20)),
         sg.InputText('', key='work_dir', tooltip=(
             'Path where to save the results. If the directory does not \n'
             'exist, it will be created.'), font=("Courier 10 Pitch", 20)),
         sg.FolderBrowse(font=("Courier 10 Pitch", 20))],
        [sg.Text('Number of cores', size=(20, 1), auto_size_text=False, justification='right',
                tooltip=('Number of cores to use to run PyCurt. Default is 0, which means \n'
                         'PyCurt will run linearly.'), font=("Courier 10 Pitch", 20)),
        sg.InputText(0, key='cores', size=(20, 10), font=("Courier 10 Pitch", 20))],
        [sg.Checkbox('Data sorting', change_submits = True, enable_events=True, size=(20, 1),
                     default=True, key='data_sorting', disabled=data_sorting, font=("Courier 10 Pitch", 20),
                     tooltip=('Whether or not to perform data sorting before curation.\n'
                     'Default is True. N.B. if this is set to False, then PyCurt expects \n'
                     'that the input folder has is the output of a previously ran data \n'
                     'sorting, i.e. the input folder cannot have a custom structure.')),
        sg.Checkbox('Renaming', change_submits = True, enable_events=True, font=("Courier 10 Pitch", 20),
                     default=False, key='renaming', disabled=data_sorting,
                     tooltip=('If "Data sorting" is selected, there is the option to rename \n'
                              'the subjects based on the information in the DICOM headers. \n'
                              'However this is not always accurate and if the field "PatientID" \n'
                              'is missing then there will be an error. Default it False.')),
        sg.Text('Subject name position', size=(25, 1), auto_size_text=False, justification='right',
                tooltip=('If "Data sorting" is selected, and "Renaming" is False, then you have \n'
                         ' to specify the position of the subject ID in the image path \n '
                         '(assuming it will be the same for all the files). For example, the \n'
                         'position in the file called /mnt/sdb/tosort/sub1/session1/image.dcm, \n'
                         'will be 3 (or -3, remember that in Python numbering starts from 0). \n'
                         'By default, it is the third position starting from the end of the path \n'
                         'i.e. -3.'), font=("Courier 10 Pitch", 20)),
        sg.InputText(-3, key='sn_pos', disabled = data_sorting,
                     size=(15, 10), font=("Courier 10 Pitch", 20))],
        [sg.Checkbox('Data curation', change_submits = True, enable_events=True, size=(20, 1),
                     default=True, key='data_curation', disabled=data_sorting, font=("Courier 10 Pitch", 20),
                     tooltip=('Whether or not to perform data curation after sorting.\n'
                     'This step converts and checks all the imaging data to NIFTI_GZ. \n')),
        sg.Checkbox('Run MR images classification', change_submits = True, enable_events=True, size=(30, 1),
                     default=True, key='mrclass', disabled=data_sorting, font=("Courier 10 Pitch", 20),
                     tooltip=('Whether or not to perform MR images classification into 6 possible \n'
                              'classes: T1, CT1, T2, FLAIR, ADC or SWI. All the images that cannot be\n'
                              'classified will be ignored.'))],
        [sg.Checkbox('Convert RT structure set', change_submits = True, enable_events=True, size=(20, 1),
                     default=True, key='extract_rts', disabled=False, font=("Courier 10 Pitch", 20),
                     tooltip=('Whether or not to extract all the contours from the RT \n'
                              'structure set.')),
        sg.Checkbox('Save rt structure with highest overlap with dose cube', change_submits = True,
                    enable_events=True, size=(55, 1),
                     default=True, key='select_roi', disabled=False, font=("Courier 10 Pitch", 20),
                     tooltip=('If True, only the RT structure with the highest overlap witht \n'
                              'dose distribution will be saved. If the dose is not found \n'
                              'then all the structures will be saved.'))],
        [sg.Checkbox('Create local database', change_submits = True, enable_events=True, size=(25, 1),
                     default=True, key='local_db', disabled=False, font=("Courier 10 Pitch", 20),
                     tooltip=('Whether or not to store the results from all the workflows in \n'
                              'one common database.'))],
        [sg.Text('Local database path', size=(25, 1), auto_size_text=False, justification='right',
                tooltip=('If "Create local database" is selected, then the path \n'
                         'where to be created, has to be provided'), font=("Courier 10 Pitch", 20)),
        sg.InputText('', key='db_path', disabled = False,
                     size=(15, 10), font=("Courier 10 Pitch", 20)),
        sg.FolderBrowse(font=("Courier 10 Pitch", 20), disabled = False, key='db_browse'),
        sg.Text('Local database project ID', size=(25, 1), auto_size_text=False, justification='right',
                tooltip=('If "Create local database" is selected, then the name \n'
                         'of the project to be created, has to be provided'), font=("Courier 10 Pitch", 20)),
        sg.InputText('', key='db_pid', disabled = False,
                     size=(15, 10), font=("Courier 10 Pitch", 20))],
        [sg.Checkbox('Save current PyCURT configuration', change_submits = True, enable_events=True, size=(35, 1),
                     default=True, key='save_conf', disabled=False, font=("Courier 10 Pitch", 20),
                     tooltip=('Whether or not to save the current PyCURT configuration to file. \n'
                              'By doing that, if something goes wrong, you can load the \n '
                              'configuration and run the same process again. The configuration \n'
                              'file will be saved in the working directory. Please note that \n'
                              'if there is a previous configuration file saved, it will be \n'
                              'overwritten. Default is True.')),
        sg.InputText('', key='save_path', disabled = False,
                     size=(15, 10), font=("Courier 10 Pitch", 20)),
        sg.FolderBrowse(font=("Courier 10 Pitch", 20), disabled = False, key='save_browse')]
        ]
    layout = [
        [sg.Text('PyCURT - Python-based automated data CUration Workflow for RadioTherapy data\n',
                 size=(80, 1), font=("Courier 10 Pitch", 24, 'bold'), justification='l')],
        [sg.Frame('Workflow parameters', input_layout, font=("Courier 10 Pitch", 22), title_color='black')],
        [sg.Text('_'  * 150)],
        [sg.Submit(font=("Courier 10 Pitch", 20)), sg.Quit(font=("Courier 10 Pitch", 20)),
         sg.Open(font=("Courier 10 Pitch", 20))]
        ]

    main_window = sg.Window('PyCurt - Main').Layout(layout)
    while True:
        main_button, values = main_window.Read()
        if main_button == 'Open':
            open_lo = [
                [sg.Text('PyCURT configuration file', size=(26, 1), auto_size_text=False,
                         justification='right', font=("Courier 10 Pitch", 20)),
                 sg.InputText(
                     '', key='in_file',
                     tooltip=('PyCURT configuration file from a previous sessions'),
                     font=("Courier 10 Pitch", 20)),
                 sg.FileBrowse(font=("Courier 10 Pitch", 20))],
                [sg.Submit(font=("Courier 10 Pitch", 20))]]
            open_window = sg.Window('Open pycurt config file').Layout(open_lo)
            config_b, config_val = open_window.Read()
            if config_b == 'Submit':
                with open(config_val['in_file'], 'rb') as f:
                    values = pickle.load(f)
                for key in values:
                    try:
                        main_window[key].Update(value=values[key])
                    except TypeError:
                        continue
                open_window.close()
                
                
        if not values['data_sorting']:
            main_window['sn_pos'].Update(disabled = True)
            main_window['sn_pos'].Update(value = -3)
            main_window['renaming'].Update(disabled = True)
        else:
            main_window['renaming'].Update(disabled = False)
            if values['renaming']:
                main_window['sn_pos'].Update(disabled = True)
            else:
                main_window['sn_pos'].Update(disabled = False)
        if not values['extract_rts']:
            main_window['select_roi'].Update(disabled = True)
            
        else:
            main_window['select_roi'].Update(disabled = False)
            
        if not values['local_db']:
            main_window['db_path'].Update(disabled = True)
            main_window['db_path'].Update(value = '')
            main_window['db_browse'].Update(disabled = True)
            main_window['db_pid'].Update(disabled = True)
            main_window['db_pid'].Update(value = '')
        else:
            main_window['db_path'].Update(disabled = False)
            main_window['db_browse'].Update(disabled = False)
            main_window['db_pid'].Update(disabled = False)
        if not values['save_conf']:
            main_window['save_path'].Update(disabled = True)
            main_window['save_path'].Update(value = '')
            main_window['save_browse'].Update(disabled = True)
        else:
            main_window['save_path'].Update(disabled = False)
            main_window['save_browse'].Update(disabled = False)
        if main_button == 'Quit':
            sys.exit()
        elif main_button == 'Submit':
            if values['save_conf']:
                if os.path.isdir(values['save_path']):
                    config_name = os.path.join(values['save_path'], 'config.pycurt')
                else:
                    config_name = values['save_path']
                with open(config_name, 'wb') as f:
                    pickle.dump(values, f)
            return values
