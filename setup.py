from setuptools import setup, find_packages

setup(name='PyCURT',
      version='1.0',
      description='Python-based automated data CUration Workflow for RadioTherapy data',
      url='https://github.com/TransRadOnc-HIT/PyCURT.git',
      python_requires='>=3.5',
      author='Francesco Sforazzini',
      author_email='f.sforazzini@dkfz.de',
      license='Apache 2.0',
      zip_safe=False,
      include_package_data=True,
      install_requires=[
      'matplotlib==3.0.2',
      'nibabel==2.3.3',
      'numpy==1.16.5',
      'pandas==0.24.0',
      'pydicom==1.2.2',
      'pynrrd==0.3.6',
      'PySimpleGUI==4.26.0',
      'torch==1.7.0',
      'torchvision==0.8.0',
      'scikit-image==0.16.2',
      'opencv-python==4.2.0.34',
      'requests==2.22.0',
      'SimpleITK==1.2.4',
      'nipype',
      'core'],
      data_files=[('resources', ['resources/random.nii.gz'])],
      dependency_links=['git+https://github.com/TransRadOnc-HIT/nipype.git@c453eac5d7efdd4e19a9bcc8a7f3d800026cc125#egg=nipype-9876543210',
                        'git+https://github.com/TransRadOnc-HIT/core.git#egg=core'],
      entry_points={
          'console_scripts': ['pycurt_gui = scripts.pycurt_gui:main',
			      'pycurt = scripts.pycurt_cmdline:main']},
      packages=find_packages(exclude=['*.tests', '*.tests.*', 'tests.*', 'tests']),
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Operating System :: Unix'
      ]
      )
