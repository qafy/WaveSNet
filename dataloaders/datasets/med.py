from __future__ import print_function, division
import os
from glob import glob
import os
import json

from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    SpatialPadd,
    DivisiblePadd,
    Resized,
)

from monai.data import CacheDataset, load_decathlon_datalist


# add the parent directory to the system path
import sys
sys.path.append(r"/Users/moritzbeckel/Desktop/WaveSNet/")
from mypath import Path

class MedDataset(CacheDataset):
    """
    Med dataset
    """

    NUM_CLASSES = 14

    def __init__(
        self,
        args,
        base_dir=Path.db_root_dir("med"),
        split="train",
    ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        self.split = split
        self.base_dir = base_dir
        self.args = args
        self.json_data = generate_json(self.base_dir)
        self.class_names = [
            "background",
            "spleen",
            "rkid",
            "lkid",
            "gall",
            "eso",
            "liver",
            "sto",
            "aorta",
            "IVC",
            "veins",
            "pancreas",
            "rad",
            "lad",
        ]

        transform = None
        datalist = None

        if self.split == "train":
            datalist = self.json_data["training"]
            print(datalist)
            transform = TRAIN_TRANSFORMS
    
        elif self.split == "val":
            datalist = self.json_data["validation"]
            transform = VAL_TRANSFORMS
     
        elif self.split == "test":
            datalist = self.json_data["testing"]
            transform = VAL_TRANSFORMS
        else:
            raise NotImplementedError("split has to be train | val | test")

        super().__init__(
            data=datalist,
            transform=transform,
            cache_num=6,
            cache_rate=1.0,
            num_workers=4,
        )

    def __str__(self):
        return "Monai_Dataset(split=" + str(self.split) + ")"

 
def generate_json(working_dir):
        """
        generate json file for the monai_dataset
        working_dir is directory file generate_json.py by default
        """
        
        print(working_dir)

        images = glob(os.path.join(working_dir, "images", "*.nii.gz"))
        #print(images)
        labels = glob(os.path.join(working_dir, "labels", "*.nii.gz"))
        #print(labels)
        testing = glob(os.path.join(working_dir, "testing", "*.nii.gz"))
        #print(testing)

        num_training = int(0.7 * len(images))
        num_validation = len(images) - num_training
        print("Amount images: " + str(len(images)))
        print("Amount for training: " + str(num_training))
        print("Amount for validation: " + str(num_validation))

        data = json.loads(JSON_SAMPLE)
        data["test"] = testing
        fit_label = [
            {"image": image, "label": " "}
            for i, image in enumerate(images[: (num_training - 1)])
        ]
        for i, label in enumerate(labels[: (num_training - 1)]):
            fit_label[i]["label"] = label
        data["training"] = fit_label

        validation_off = num_validation + num_training - 1
        fit_label = [
            {"image": image, "label": " "}
            for i, image in enumerate(images[num_training:validation_off])
        ]
        for i, label in enumerate(labels[num_training:validation_off]):
            fit_label[i]["label"] = label
        data["validation"] = fit_label
        
        return data

TRAIN_TRANSFORMS = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            Resized(
                keys=["image", "label"],
                spatial_size=(96, 96, 96),
                mode=["linear", "nearest"],
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]
    )

VAL_TRANSFORMS = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            Resized(
                keys=["image", "label"],
                spatial_size=(96, 96, 96),
                mode=["linear", "nearest"],
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]
    )

JSON_SAMPLE = """
    {
        "description": "btcv yucheng",
        "labels": {
            "0": "background",
            "1": "spleen",
            "2": "rkid",
            "3": "lkid",
            "4": "gall",
            "5": "eso",
            "6": "liver",
            "7": "sto",
            "8": "aorta",
            "9": "IVC",
            "10": "veins",
            "11": "pancreas",
            "12": "rad",
            "13": "lad"
        },
        "licence": "yt",
        "modality": {
            "0": "CT"
        },
        "name": "btcv",
        "numTest": 20,
        "numTraining": 80,
        "reference": "Vanderbilt University",
        "release": "1.0 06/08/2015",
        "tensorImageSize": "3D",
        "test": [],
        "training": [],
        "validation": [

            {
                "image": "imagesTr/img0040.nii.gz",
                "label": "labelsTr/label0040.nii.gz"
            }
        ]
    }
    """


import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

class MedDataset_2D(Dataset):
    
    NUM_CLASSES = 14
    
    def __init__(self, nifti_files, axis=2):
        """
        Args:
            nifti_files (list): List of file paths to NIfTI files.
            axis (int): The axis along which to extract 2D slices. Default is 2 (axial).
        """
        
        self.split = split
        self.base_dir = base_dir
        self.args = args
        self.json_data = generate_json(self.base_dir)
        self.class_names = [
            "background",
            "spleen",
            "rkid",
            "lkid",
            "gall",
            "eso",
            "liver",
            "sto",
            "aorta",
            "IVC",
            "veins",
            "pancreas",
            "rad",
            "lad",
        ]

        transform = None
        datalist = None

        if self.split == "train":
            datalist = self.json_data["training"]
            print(datalist)
            transform = TRAIN_TRANSFORMS
    
        elif self.split == "val":
            datalist = self.json_data["validation"]
            transform = VAL_TRANSFORMS
     
        elif self.split == "test":
            datalist = self.json_data["testing"]
            transform = VAL_TRANSFORMS
        else:
            raise NotImplementedError("split has to be train | val | test")

        super().__init__(
            data=datalist,
            transform=transform,
            cache_num=6,
            cache_rate=1.0,
            num_workers=4,
        )
        self.nifti_files = nifti_files
        self.axis = axis
        self.index_mapping = []
        
        # Calculate the total number of 2D slices and map them to respective NIfTI files
        total_slices = 0
        for nifti_file in nifti_files:
            img = nib.load(nifti_file)
            num_slices = img.shape[axis]
            for slice_idx in range(num_slices):
                self.index_mapping.append((nifti_file, slice_idx))
    
    def __len__(self):
        return len(self.index_mapping)
    
    def __getitem__(self, idx):
        nifti_file, slice_idx = self.index_mapping[idx] 
        img = nib.load(nifti_file).get_fdata()
        if self.axis == 0:
            return img[slice_idx, :, :]
        elif self.axis == 1:
            return img[:, slice_idx, :]
        else:
            return img[:, :, slice_idx]

# # Example usage:
# nifti_files = ['/path/to/nifti1.nii', '/path/to/nifti2.nii']
# dataset = NiftiTo2DSlicesDataset(nifti_files, axis=2)

# # Get the first 2D slice
# first_slice = dataset[0]

if __name__ == "__main__":
    import numpy as np
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = MedDataset_2D(args, split="train")

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=8)

    # slice_map = {
    # "DET0000101_avg.nii.gz": 20, 
    # "DET0016101_avg.nii.gz": 20,
    # "DET0004701_avg.nii.gz": 20,
    # }
    # case_num = 0
    # img_name = os.path.split(voc_train[case_num]["image"].meta["filename_or_obj"])[1]
    # img = voc_train[case_num]["image"]
    # label = voc_train[case_num]["label"]
    # img_shape = img.shape
    # label_shape = label.shape
    # print(f"image shape: {img_shape}, label shape: {label_shape}")
    # plt.figure("image", (18, 6))
    # plt.subplot(1, 2, 1)
    # plt.title("image")
    # plt.imshow(img[0, :, :, slice_map[img_name]].detach().cpu(), cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.title("label")
    # plt.imshow(label[0, :, :, slice_map[img_name]].detach().cpu())
    # plt.show()
    
    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample["image"].numpy()
            print(img.shape)
            gt = sample["label"].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset="pascal")
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title("display")
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
