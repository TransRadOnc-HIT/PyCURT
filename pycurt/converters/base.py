from core.utils.filemanip import split_filename

class BaseConverter(object):
    
    def __init__(self, toConvert, clean=False, bin_path=''):
        print('Started image format conversion...')
        self.basedir, self.filename, _ = split_filename(toConvert)
        self.filename = self.filename.split('.')[0]
        self.toConvert = toConvert
        self.clean = clean
        self.bin_path = bin_path
    
    def convert(self):

        raise NotImplementedError