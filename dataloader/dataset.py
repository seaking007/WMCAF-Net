import argparse
import os
import cv2
from torch.utils.data import Dataset
import os
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MedicalDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        transform=None,
        train_file_dir="train.txt",
        val_file_dir="val.txt",
        test_file_dir="test.txt",
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.train_list = []
        self.semi_list = []

        if self.split == "train":
            with open(os.path.join(self._base_dir, train_file_dir), "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(os.path.join(self._base_dir, val_file_dir), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "test":
            with open(os.path.join(self._base_dir, test_file_dir), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {}  {} samples".format(len(self.sample_list), self.split))

    def __len__(self):
        return len(self.sample_list)

    # def __getitem__(self, idx):
    #
    #     case = self.sample_list[idx]
    #     # print("case",case)
    #
    #     # image_path = os.path.join(self._base_dir, 'images', case + '.png')
    #     # 打印图像路径
    #     # print(f"Image path: {image_path}")
    #     image = cv2.imread(os.path.join(self._base_dir, 'images', case + '.png'))
    #
    #     # image = cv2.imread(image_path)
    #     # print(case)
    #     mask_path=os.path.join(self._base_dir,'masks','0',case+'.png')
    #     # print("mask_path",mask_path)
    #     if not os.path.exists(mask_path):
    #         raise FileNotFoundError(f"Image file not found: {mask_path}")
    #     label = cv2.imread(os.path.join(self._base_dir, 'masks', '0', case + '.png'), cv2.IMREAD_GRAYSCALE)[..., None]
    #
    #
    #     # label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #
    #
    #     augmented = self.transform(image=image, mask=label)
    #
    #     image = augmented['image']
    #     label = augmented['mask']
    #
    #     image = image.astype('float32') / 255
    #     image = image.transpose(2, 0, 1)
    #
    #     label = label.astype('float32') / 255
    #     label = label.transpose(2, 0, 1)
    #     sample = {"image": image, "label": label, "case": case}
    #     return sample

    def __getitem__(self, idx):
        case = self.sample_list[idx].strip()

        # 如果 txt 里没有后缀，就补 .png；如果已经有 .png，就不重复补
        if not case.endswith(".png"):
            filename = case + ".png"
        else:
            filename = case

        image_path = os.path.join(self._base_dir, "images", filename)
        mask_path = os.path.join(self._base_dir, "masks", "0", filename)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        image = cv2.imread(image_path)
        label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"cv2.imread failed to read image: {image_path}")

        if label is None:
            raise RuntimeError(f"cv2.imread failed to read mask: {mask_path}")

        label = label[..., None]

        if self.transform is not None:
            augmented = self.transform(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"]

        image = image.astype("float32") / 255.0
        image = image.transpose(2, 0, 1)

        label = label.astype("float32") / 255.0
        label = label.transpose(2, 0, 1)

        sample = {
            "image": image,
            "label": label,
            "case": case
        }

        return sample




