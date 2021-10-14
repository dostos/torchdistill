# No 'default_generator' in torch/__init__.pyi
from torchvision.datasets.vision import VisionDataset
from typing import TypeVar, Dict

T = TypeVar('T')

def constant(f):
    def fset(self, value):
        raise TypeError
    def fget(self):
        return f()
    return property(fget, fset)

# TODO: May need better super class that returns sample, target from __getitem__ by default
class CrossDatasetAdapter(VisionDataset):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        target_to_source (dict): conversion from target label to source label
    """
    dataset: VisionDataset
    target_to_source: Dict[str, str]
    
    OutOfSourceLabel=-1

    def __init__(self, dataset: VisionDataset, target_to_source: Dict[str, str]) -> None:
        self.dataset = dataset
        self.target_to_source = dict()
        self.sources = set()
        
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        
        for k,v in target_to_source.items():
            self.target_to_source[int(k)] = int(v)
            self.sources.add(v)
        
        for class_name in self.classes:
            if self.class_to_idx[class_name] not in self.sources:
                self.dummy = self.class_to_idx[class_name]
                break

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        return sample, self.target_to_source[target] if target in self.target_to_source else self.OutOfSourceLabel

    def __len__(self):
        return len(self.dataset)