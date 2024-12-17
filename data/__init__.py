from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torchvision import transforms
from utils.tools import RandCrop, RandHorizontalFlip, RandRotate, ToTensor
import cv2
import numpy as np
import os

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)


class DIV2KDataset(Dataset):
    def __init__(self, dir_data, transform,scale):
        super(DIV2KDataset, self).__init__()
        self.dir_data = dir_data
        self.transform = transform
        self.dirname_HR = os.path.join(self.dir_data, 'DIV2K/DIV2K_train_HR')
        self.dirname_LR = os.path.join(self.dir_data, 'DIV2K/DIV2K_train_LR_bicubic/X') + str(scale[0])


        self.filelist_LR = os.listdir(self.dirname_LR)
        self.filelist_LR.sort()
        self.filelist_HR = os.listdir(self.dirname_HR)
        self.filelist_HR.sort()

    def __len__(self):
        return len(self.filelist_LR)

    def __getitem__(self, idx):
        img_name_LR = self.filelist_LR[idx]
        img_LR = cv2.imread(os.path.join(self.dirname_LR, img_name_LR), cv2.IMREAD_COLOR)
        img_LR = cv2.cvtColor(img_LR, cv2.COLOR_BGR2RGB)
        img_LR = np.array(img_LR).astype('float32') / 255

        img_name_HR = self.filelist_HR[idx]
        img_HR = cv2.imread(os.path.join(self.dirname_HR, img_name_HR), cv2.IMREAD_COLOR)
        img_HR = cv2.cvtColor(img_HR, cv2.COLOR_BGR2RGB)
        img_HR = np.array(img_HR).astype('float32') / 255

        sample = {'img_LR': img_LR, 'img_HR': img_HR}

        if self.transform:
            sample = self.transform(sample)

        return sample
    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)


class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            train_dataset = DIV2KDataset(
                dir_data=args.dir_data,
                transform=transforms.Compose(
                    [RandCrop(args.patch_size, args.scale), RandHorizontalFlip(), RandRotate(), ToTensor()]),
                scale=args.scale
            )

            self.loader_train = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                drop_last=True,
                shuffle=True
            )

        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)
            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )

