import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset


class SSDataset(Dataset):
    """
    AVM system SS (Semantic Segmentation) dataset
    """

    all_colors = []

    def __init__(self, dataset_path, is_train):
        self.is_train = is_train
        self.dataset_path = dataset_path
        self.test_db = self._path('test_db.txt')
        self.train_db = self._path('train_db.txt')

        with open(self.train_db if is_train else self.test_db) as db_file:
            db_file = db_file.read()

        lines = [line for line in db_file.split('\n') if line.count(' ') >= 1]
        image_list, gt_list = [], []
        for line in lines:
            sp = line.split(' ')
            image_list.append(sp[0].strip())
            gt_list.append(sp[1].strip())

        self.dataset = []
        for idx, (db_image, db_gt) in enumerate(zip(image_list, gt_list)):
            db_image_path, db_gt_path = self._path(db_image), self._path(db_gt)
            db_image, db_gt = self._read_image(db_image_path), self._read_gt(db_gt_path)
            self.dataset.append(((db_image_path, db_image), (db_gt_path, db_gt)))

            if (idx + 1) % 100 == 0:
                print('Reading dataset: {}/{}'.format(idx + 1, len(lines)))

        self._parse_gt()  # images: (3, w, h), gts: (ch, w, h)
        print(('Train' if is_train else 'Test') + ' dataset: image count = {}'.format(self.__len__()))

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def _path(self, name):
        if name[0] == '/':
            name = name[1:]
        return os.path.join(self.dataset_path, name)

    def _read_image(self, image_path):
        """
        Read dataset/images/00000000.jpg, transpose to c,w,h
        """
        image = Image.open(image_path)
        image = np.array(image)  # h w c
        image = image.transpose((2, 1, 0))  # c w h
        image = image / 255  # normalization
        return image.astype(np.float)

    def _read_gt(self, image_path):
        """
        Read dataset/gt/00000000.jpg, transpose to w,h,c
        """
        image = Image.open(image_path)
        image = np.array(image)  # h w c
        image = image.transpose((1, 0, 2))  # w h c
        return image.astype(np.float)

    def _parse_gt(self):
        """
        Parse a list of gt images, get all_colors list, and split class to different feature channels
        """
        # get final all_colors list
        if len(SSDataset.all_colors) == 0:
            target_colors = []
            target_path = ''
            for _, (_, gt_image) in enumerate(self.dataset):
                gt_image_path = gt_image[0]
                gt_image = gt_image[1]
                image_w, image_h = gt_image.shape[0:2]
                pixels = [tuple(a) for b in gt_image.tolist() for a in b]
                current_colors = list(set(pixels))

                # Each feature color must have more than 2% pixels
                current_colors = list(filter(lambda color: color in target_colors or pixels.count(color) > len(pixels) * 0.02, current_colors))
                if len(current_colors) > len(target_colors):
                    target_colors = current_colors
                    target_path = gt_image_path

            SSDataset.all_colors = target_colors
            print('Using {} as mask colors'.format(gt_image_path))
            print('Mask colors: {}'.format(target_colors))

        # parse dataset tuple list through all_colors
        for line_idx, (image, gt_image) in enumerate(self.dataset):
            image = image[1]
            gt_image = gt_image[1]
            image_w, image_h = gt_image.shape[0:2]
            color_pixels = {}
            for w in range(image_w):
                for h in range(image_h):
                    color = tuple(gt_image[w][h])
                    if color not in SSDataset.all_colors:
                        continue
                    if color not in color_pixels:
                        color_pixels[color] = []
                    color_pixels[color].append((w, h))

            gts = np.zeros((len(SSDataset.all_colors), image_w, image_h)).astype(np.float)
            for idx, color in enumerate(SSDataset.all_colors):
                if color in color_pixels:
                    for (x, y) in color_pixels[color]:
                        gts[idx][x][y] = 1.

            image = torch.from_numpy(image).type(torch.FloatTensor)
            gts = torch.from_numpy(gts).type(torch.FloatTensor)
            self.dataset[line_idx] = (image, gts)


def load_test_image(path):
    image = Image.open(path)
    image = np.array(image)  # h w c
    image = image.transpose((2, 1, 0))  # c w h
    image = image[np.newaxis, :]
    image = (image / 255).astype(np.float)  # normalization
    image = torch.tensor(image).type(torch.FloatTensor)
    return image
