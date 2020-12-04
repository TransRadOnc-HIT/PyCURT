from nipype.interfaces.base import (
    TraitedSpec, traits, File, CommandLineInputSpec, CommandLine,
    Directory)
import os.path as op
from nipype.interfaces.base import isdefined


class DoseConverterInputSpecs(CommandLineInputSpec):
    
    input_dose = File(mandatory=True, exists=True, argstr='--input %s',
                      desc='Dose DICOM file.')
    out_name = traits.Str('dose.nii.gz', usedefault=True,
                          argstr='--output-dose-img %s', desc='Output name. '
                          'Default is dose.nii.gz.')


class DoseConverterOutputSpec(TraitedSpec):
    
    out_file = File(exists=True, desc='Converted dose file.')


class DoseConverter(CommandLine):
    
    _cmd = 'plastimatch convert'
    input_spec = DoseConverterInputSpecs
    output_spec = DoseConverterOutputSpec
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.out_name):
            outputs['out_file'] = op.abspath(self.inputs.out_name)

        return outputs
    

class RTStructureCoverterInputSpec(CommandLineInputSpec):
    
    input_ss = File(mandatory=True, exists=True, argstr='--input %s',
                    desc='Structure set DICOM file.')
    reference_ct = File(mandatory=True, exists=True, argstr='--referenced-ct %s',
                        desc='Reference CT image file.')
    out_prefix = traits.Str(
        'rs_structures', usedefault=True, argstr='--output-prefix %s',
        desc='Name of the folder that will be created containing all the RT structures. '
        'Default is rs_structures.')


class RTStructureCoverterOutputSpec(TraitedSpec):
    
    out_structures = Directory(exists=True, desc='Directory with converted RT structures.')


class RTStructureCoverter(CommandLine):

    _cmd = 'plastimatch convert'
    input_spec = RTStructureCoverterInputSpec
    output_spec = RTStructureCoverterOutputSpec
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.out_prefix):
            outputs['out_structures'] = op.abspath(self.inputs.out_prefix)

        return outputs
    