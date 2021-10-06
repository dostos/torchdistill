from __future__ import print_function
from PIL import Image

import os
import os.path
import fnmatch
from typing import Any, Callable, List, Optional, Union, Tuple

from imageio import imread

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity, download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.caltech import Caltech256

from torchdistill.datasets.registry import register_dataset


@register_dataset
class CustomCaltech256(VisionDataset):
    """`Caltech 256 <http://www.vision.caltech.edu/Image_Datasets/Caltech256/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``caltech256`` exists or will be saved to if download is set to True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(CustomCaltech256, self).__init__(
            os.path.join(root, "caltech256"), transform=transform, target_transform=target_transform
        )
        os.makedirs(self.root, exist_ok=True)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." + " You can use download=True to download it")

        self.categories = sorted(os.listdir(os.path.join(self.root, "256_ObjectCategories")))
        self.classes = [class_name[4:] for class_name in self.categories]
        self.index: List[int] = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(fnmatch.filter(os.listdir(os.path.join(self.root, "256_ObjectCategories", c)), "*.jpg"))
            self.index.extend(range(0, n))
            self.y.extend(n * [i])
            print(i)
            
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(
            os.path.join(
                self.root,
                "256_ObjectCategories",
                self.categories[self.y[index]],
                "{:03d}_{:04d}.jpg".format(self.y[index] + 1, self.index[index] + 1),
            )
        ).convert('RGB')

        target = self.y[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "256_ObjectCategories"))

    def __len__(self) -> int:
        return len(self.index)

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar",
            self.root,
            filename="256_ObjectCategories.tar",
            md5="67b4f42ca05d46448c6bb8ecd2220f6d",
        )
