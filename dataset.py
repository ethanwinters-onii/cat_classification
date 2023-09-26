import os
import pickle
from traceback import print_exception

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

class DentDataset(ImageFolder):
    def __init__(self, 
                root, 
                train=True, 
                year=2022, 
                transform=None, 
                target_transform=None,
                num_classes = 1000,
                option='train',
                loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        self.nb_classes = num_classes
        self.option = option

        self.samples = []
        with open('dataset_24_09_2023.pkl', 'rb') as file:
            train_x, val_x, train_y, val_y = pickle.load(file)
        
        if self.option == 'train':
            for idx in range(len(train_x)):
                self.samples.append((train_x[idx], train_y[idx]))
        elif self.option == 'val':
            for idx in range(len(val_x)):
                self.samples.append((val_x[idx], val_y[idx]))
        else:
            print('Error to load data.')
        
    # __getitem__ and __len__ inherited from ImageFolder
        


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        print('Train transform')
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            hflip=0.,
            vflip=0.,
            re_count=args.recount,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )

        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4
            )

        return transform
    
    print('Validation transform')
    t = []
    if resize_im:
        # size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize((224,224)),  # to maintain same ratio w.r.t. 224 images
        )
        # t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_dataset(is_train, args):
    transforms = build_transform(is_train, args)

    option = 'train' if is_train else 'val'
    dataset = DentDataset(args.data_path, train=is_train, transform=transforms, num_classes=args.classes, option=option)
    nb_classes = dataset.nb_classes

    return dataset, nb_classes
        
    