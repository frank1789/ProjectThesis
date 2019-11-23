#!usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import errno
import os
import random
import re
import shutil
import sys
import time

from tqdm import tqdm


class PrepareDataset(object):
    """
    It is a class that allows to prepare the analysis of the dataset once imported from a source.
    Inside there are methods for manipulating the data-set.
                                           _
                                          | \_____
                                          | data  |
                                          |_______|
                                            |
                            +---------------+-------------------+
                            |               |                   |
                           _              _                    _
                          | \_____       | \_________         | \_____
                          | train |      | validate  |        | test  |
                          |_______|      |___________|        |_______|
    """

    # extensions of the files to be ignored, such as hidden files
    _exclude_ext = ['.DS_Store', 'desktop.ini', 'Desktop.ini', '.csv', '.json', '.h5']

    def __init__(self, path) -> None:
        self._default_train = os.path.join(path, "train")  # folder contains the data-set and his sub-folder
        self._default_validate = os.path.join(path, "validate")  # folder contains the data-set and his sub-folder
        self._default_test = os.path.join(path, "test")  # folder contains test-set

        if not os.path.exists(self._default_train):
            # make train folder
            os.makedirs(self._default_train)
            os.makedirs(os.path.join(self._default_train, "annotations"))
            print("==> Generate train folder created at: ", self._default_train)
        if not os.path.exists(self._default_validate):
            # make validate folder
            os.makedirs(self._default_validate)
            os.makedirs(os.path.join(self._default_validate, "annotations"))
            print("==> Generate train folder created at: ", self._default_validate)
        if not os.path.exists(self._default_test):
            # make test folder
            os.makedirs(self._default_test)
            print("==> Generate test folder created at: ", self._default_test)

    def _scan_files(self, path) -> list:
        """
        Scans the folder by discarding all files that have an unsupported extension.
        :param path (str) folder's path
        :return list_files (list) list contains files
        """
        list_files = []
        if os.path.exists(path):  # check if is valid path
            for root, dirname, files in os.walk(path):
                for namefile in files:
                    list_files.append(os.path.join(root, namefile))  # fill the lst with all files
        else:
            print(FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path))

        return list_files

    def copy_file(self, source, destination) -> None:
        pass


class DataSet(PrepareDataset):
    def __init__(self, path_dataset, split_train_validate=30):
        """
        Default constructor of data-set. Once invoked it checks the validity of the examined folder, proceeds in the
        generation of the list of the files of the dataset and of the relative classification.

        Parameters
        ----------
        :param path_dataset: (str) path's data-set.
        """
        super(DataSet, self).__init__(path_dataset)
        # check if folder exist, then build the database
        if os.path.exists(path_dataset) and os.path.isdir(path_dataset):
            self.files = self._scan_files(path_dataset)
            self.validate_files = self._make_validate_dir(split_train_validate)
        else:
            print(FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_dataset))

    def _make_validate_dir(self, split_train_validate=30) -> list:
        """
        Copy from original folder and split in two folder train and validate
                            data
                             |
                    +--------+--------+
                    |                 |
                  train            validate

        Parameters
        ----------
        :param split_train_validate: (int) indicates how much to divide the training set
        """
        source = []
        if 0 < split_train_validate <= 100:
            split_train_validate /= 100
            imgs_filtered = list(filter(lambda x: x.endswith("jpg"), self.files))
            annotations_filtered = list(filter(lambda x: x.endswith("xml"), self.files))
            # sort lists
            imgs_filtered.sort()
            annotations_filtered.sort()
            # join in list of tuple
            files = list(zip(imgs_filtered, annotations_filtered))
            # Amount of random files you'd like to select
            random_amount = int((len(imgs_filtered) * split_train_validate) + 1)
            mess = "==> Total images {},".format(len(imgs_filtered))
            mess += "\tsplit to {:3d}%,".format(int(split_train_validate * 100))
            mess += "\tfiles in validate folder: {}".format(random_amount)
            print(mess)
            for img in range(random_amount):
                source.append(random.choice(files))
            source.sort()
            out = list(sum(source, ()))
            return out
        else:
            raise ValueError("train_split_validate must be a number to divide the training set must be 0 and 100")

    def copy_file(self, source, destination) -> None:
        """
        Copy images and annotations in correct folder train and validate.

        Parameters
        ----------
        :param source: (list) contains the path of each file in the dataset.
        :param destination: (str) contains the destination path.
        """
        print("==> Start copy files in folders:")
        count_imgs = 0
        count_annotations = 0
        for file in tqdm(source):
            match = re.search(r"\w+.jpg|\w+.xml", file)
            if match is not None:
                filename = match.group(0)
                if filename.endswith("jpg"):
                    shutil.copy(file, os.path.join(destination, filename))
                    count_imgs += 1
                if filename.endswith("xml"):
                    shutil.copy(file, os.path.join(destination, "annotations", filename))
                    count_annotations += 1
        print("\n==> Done")
        print("==> Copied {} images in {} folders.".format(count_imgs, destination))
        print("==> Copied {} annotation in {} folders.".format(count_annotations, destination))
        print("-" * 79, end='\n')

    def elaborate(self) -> None:
        """
        Perform operation on dataset.
        """
        self.copy_file(self.files, self._default_train)
        time.sleep(0.5)
        self.copy_file(self.validate_files, self._default_validate)


if __name__ == '__main__':
    # parsing argument script
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', action='store', dest='rawdataset', help='Original folder raw dataset')
    parser.add_argument('-s', '--split', action='store', dest='split', default=30,
                        help='Split percentage train and validate set.')

    args = parser.parse_args()
    # process data set
    if args.split != 30:
        out_dataset = DataSet(args.rawdataset, int(args.split))
    else:
        out_dataset = DataSet(args.rawdataset, int(args.split))
    out_dataset.elaborate()
    sys.exit()
