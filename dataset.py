import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset


class SSDataset(Dataset):
    """
    AVM system SS (Semantic Segmentation) dataset
    """

    def __init__(self, dataset_path, is_train):
        self.is_train = is_train
        self.dataset_path = dataset_path
        self.test_db = self._path('test_db.txt')
        self.train_db = self._path('train_db.txt')

        with open(self.train_db if is_train else self.test_db) as db_file:
            db_file = db_file.read()

        self.dataset = []
        for db_line in db_file.split('\n'):
            db_image, db_gt = db_line.split(' ')[:2]
            db_image, db_gt =  self._path(db_image), self._path(db_gt)
            print(db_image, db_gt)
            db_image = self._read_image(db_image)
            db_gt = self._read_image(db_gt)
            self.dataset.append((db_image, db_gt))

        print(('Train' if is_train else 'Test') + 'dataset: image count = {}'.format(self.__len__()))

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def _path(self, name):
        if name[0] == '/':
            name = name[1:]
        return os.path.join(self.dataset_path, name)

    def _read_image(self, image_path):
        image = Image.open(image_path)
        image = np.array(image)  # w h c
        image = image.transpose((2, 0, 1))  # c w h
        return image / 255  # normalization
