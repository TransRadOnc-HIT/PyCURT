# PyCURT: Python-based automated data CUration Workflow for RadioTherapy data
In the field of "Big Data" analysis in medical imaging, the problem of data curation is one of the major bottleneck, especially when data from multiple sites have to be collected and processed. Data structure, imaging protocols, scanners and nomenclature are often incosistent. Furthermore, in the field of radiation oncology, patients undergo several imaging sessions and have data coming from different modalities, like MRI, CT, PET and Radiotherapy. For this reason, it is extremely hard to have a workflow that can automatically curate this type data.
With PyCURT, we are providing a tool that can automatically sort any pertinent information from an imaging study. It was originally develop to collect radiotherapy information from imaging studies but it is now also able to label MR images into 6 different contrasts (T1 pre and post contrast agent injection, T2, FLAIR, ADC and SWI) thanks to MRClass (REF).

PyCURT has been thought as general as possible. It can take as input a folder that contains DICOM files in any strcuture. It will go through the folder and collect all the DICOM files. It will then check their integrity and it will extract important parameters, like study date, patient ID and so on. It will also look for the radiotherapy plan, that will be then used to link together all the radiotherapy data (like planning CT, structure set and dose cubes). At the end, there will be a folder with a subject/session/scan structure, and the session with RT data will be labelled as RT_* . All the other sessions will contain images coming from MR or CT modalties. All the MR images will be labelled as one of the 6 contrast mentioned before (N.B. right now, all the MR contrasts different from those listed before will be ignored and not copied in the final folder. However they will be available in the cache directory).
After that, imaging data can be converted to Nifti, which is a format often used in subsequent analyses. MR and CT images, as well as dose cubes are converted. There is also the option to extract all the manually contoured structures from the RT structure set and save them as Nifti. Furthermore, instead of saving all of the strucures, you can choose to save only the one with the highest overlap with used dose cube, if present.
Finally, you have the option to create a local database where all the outputs are saved in a consistent and easy-to-handle way.

# Installation
PyCURT supports only Python>3.5 and it has been tested only on Linux (Ubuntu) platforms.
We very strongly recommend you install PyCURT in a virtual environment. [Here is a quick how-to for Ubuntu](https://linoxide.com/linux-how-to/setup-python-virtual-environment-ubuntu/). [Conda enviroments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) can also be used.
Before installing PyCURT, there are two external tools that need to be installed:
1. [dcm2niix](https://github.com/rordenlab/dcm2niix/releases/tag/v1.0.20200331) which is required to convert imaging data from DICOM to Nifti. Following the link, you will be prompted to download the pre-compiled version (choose Linux os).
2. [plastimatch](https://www.plastimatch.org/) which is required to process RT data. You can download pre-compiled binaries or build it from source.

Once those two tools are installed, you can install PyCURT. You can do it by typing the following few steps in a terminal:
1. Clone this repository, `git clone https://github.com/TRO-HIT/PyCURT.git`.
2. cd into the cloned directory, `cd PyCURT`.
3. Create a virtual (or conda) environment. With anaconda you can do it by typing `conda create -n pycurt python=3.7`.
4. Activate conda environment, `conda activate pycurt`.
5. Install PyCURT by typing `python setup.py install`

Last step will create two commands, `pycurt` and `pycurt_gui`. The first one can be used to run PyCURT from command line, the second one will open a graphical user interface that can be used to configure PyCURT.
