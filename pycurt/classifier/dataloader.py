from skimage.transform import resize
import numpy as np
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.dataloading.data_loader import DataLoader


class BpClassDataLoader2D(DataLoader):

    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded,
                 seed_for_shuffle=1234, return_incomplete=False, shuffle=True, crop=False):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)
        patch_size is the spatial size the retured batch will have
        """
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle,
                         return_incomplete, shuffle, False)
        self.patch_size = patch_size
        self.num_modalities = 1
        self.indices = list(range(len(data)))
        self.crop = crop

    def generate_train_batch(self):

        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        data = np.zeros((self.batch_size, self.num_modalities, *self.patch_size), dtype=np.float32)

        for i, _ in enumerate(patients_for_batch):
            patient_data = patients_for_batch[i]
            if not self.crop:
                patient_data = resize(patient_data, (patient_data.shape[0],)+self.patch_size,
                                      anti_aliasing=True)
                data[i] = patient_data
            else:

                # this will only pad patient_data if its shape is smaller than self.patch_size
                patient_data = pad_nd_image(patient_data, self.patch_size)

                # now random crop to self.patch_size

                patient_data = crop(patient_data[None], crop_size=self.patch_size, crop_type="center")
                data[i] = patient_data[0]

        return {'data': data}
