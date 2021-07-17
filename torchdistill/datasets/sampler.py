import bisect
import copy
import multiprocessing as mp
import parmap

from collections import defaultdict

import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
from torch.utils.data.sampler import WeightedRandomSampler, BatchSampler, Sampler
from torch.utils.model_zoo import tqdm

from torchdistill.common.constant import def_logger
from torchdistill.datasets.wrapper import BaseDatasetWrapper

logger = def_logger.getChild(__name__)
BATCH_SAMPLER_CLASS_DICT = dict()


def register_batch_sampler_class(cls):
    BATCH_SAMPLER_CLASS_DICT[cls.__name__] = cls
    return cls


@register_batch_sampler_class
class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, sampler, group_ids, batch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                'sampler should be an instance of '
                'torch.utils.data.Sampler, but got sampler={}'.format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            # for the remaining batches, take first the buffers with largest number
            # of elements
            for group_id, _ in sorted(buffer_per_group.items(),
                                      key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                buffer_per_group[group_id].extend(
                    samples_per_group[group_id][:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler) // self.batch_size


class _SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def _compute_aspect_ratios_slow(dataset, indices=None):
    logger.info('Your dataset doesn\'t support the fast path for '
                'computing the aspect ratios, so will iterate over '
                'the full dataset and load every image instead. '
                'This might take some time...')
    if indices is None:
        indices = range(len(dataset))

    sampler = _SubsetSampler(indices)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=sampler,
        num_workers=14,  # you might want to increase it for faster processing
        collate_fn=lambda x: x[0])
    aspect_ratios = []
    with tqdm(total=len(dataset)) as pbar:
        for _i, tuple_item in enumerate(data_loader):
            img = tuple_item[0]
            pbar.update(1)
            height, width = img.shape[-2:]
            aspect_ratio = float(width) / float(height)
            aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_custom_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        height, width = dataset.get_height_and_width(i)
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_coco_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        img_info = dataset.coco.imgs[dataset.ids[i]]
        aspect_ratio = float(img_info['width']) / float(img_info['height'])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_voc_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        # this doesn't load the data into memory, because PIL loads it lazily
        width, height = Image.open(dataset.images[i]).size
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_subset_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))

    ds_indices = [dataset.indices[i] for i in indices]
    return compute_aspect_ratios(dataset.dataset, ds_indices)


def compute_aspect_ratios(dataset, indices=None):
    target_dataset = dataset.org_dataset if isinstance(dataset, BaseDatasetWrapper) else dataset
    if hasattr(target_dataset, 'get_height_and_width'):
        return _compute_aspect_ratios_custom_dataset(target_dataset, indices)

    if isinstance(target_dataset, torchvision.datasets.CocoDetection):
        return _compute_aspect_ratios_coco_dataset(target_dataset, indices)

    if isinstance(target_dataset, torchvision.datasets.VOCDetection):
        return _compute_aspect_ratios_voc_dataset(target_dataset, indices)

    if isinstance(target_dataset, torch.utils.data.Subset):
        return _compute_aspect_ratios_subset_dataset(target_dataset, indices)

    # slow path
    return _compute_aspect_ratios_slow(target_dataset, indices)


def _quantize(x, bins):
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def create_aspect_ratio_groups(dataset, k=0):
    aspect_ratios = compute_aspect_ratios(dataset)
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]
    groups = _quantize(aspect_ratios, bins)
    # count number of elements per group
    counts = np.unique(groups, return_counts=True)[1]
    fbins = [0] + bins + [np.inf]
    logger.info('Using {} as bins for aspect ratio quantization'.format(fbins))
    logger.info('Count of instances per bin: {}'.format(counts))
    return groups


def get_batch_sampler(dataset, class_name, *args, **kwargs):
    if class_name not in BATCH_SAMPLER_CLASS_DICT and class_name != 'BatchSampler':
        logger.info('No batch sampler called `{}` is registered.'.format(class_name))
        return None

    batch_sampler_cls = BatchSampler if class_name == 'BatchSampler' else BATCH_SAMPLER_CLASS_DICT[class_name]
    if batch_sampler_cls == GroupedBatchSampler:
        group_ids = create_aspect_ratio_groups(dataset, k=kwargs.pop('aspect_ratio_group_factor'))
        return batch_sampler_cls(*args, group_ids, **kwargs)
    return batch_sampler_cls(*args, **kwargs)


def count(indices, dataset):
    global single 
    worker_category_counts = dict()

    for category in dataset.coco.cats.keys():
        worker_category_counts[category] = 0

    for i in indices:
        [img, result] = dataset.get_raw(i)
        for label in result['annotations']:
            worker_category_counts[label['category_id']] = worker_category_counts[label['category_id']] + 1
    
    return worker_category_counts

class TargetClassSampler(WeightedRandomSampler):
    def __init__(self, dataset, target_classes, target_weight = 0.5):
        if isinstance(dataset, torch.utils.data.Subset):
            subset = dataset
            dataset = dataset.dataset

            category_counts = dict()
            category_weights = dict()
            total_count = 0
            for category in dataset.coco.cats.keys():
                category_counts[category] = 0

            num_cores = mp.cpu_count()
            result = parmap.map(count, np.array_split(subset.indices, num_cores), dataset=dataset, pm_pbar=True, pm_processes=num_cores)

            for category in dataset.coco.cats.keys():
                for worker_category_counts in result: 
                    category_counts[category] = category_counts[category] + worker_category_counts[category]
                    total_count = total_count + worker_category_counts[category]
            
            num_target_classes = len(target_classes)
            num_classes = len(dataset.coco.cats.keys())

            for category in dataset.coco.cats.keys():
                if category in target_classes:
                    category_weights[category] = target_weight
                else:
                        



            sample_weights = [1 / len(subset)] * len(subset)
        else:
            target_classes_set = set()
            idx2class = {v: k for k, v in dataset.class_to_idx.items()}

            for target_class in target_classes:
                if isinstance(target_class, int) and target_class in idx2class:
                    target_classes_set.add(target_class)
                elif isinstance(target_class, str) and target_class in dataset.class_to_idx:
                    target_classes_set.add(dataset.class_to_idx[target_class])

            target_classes = target_classes_set
            num_target_classes = len(target_classes)
            num_classes = len(dataset.classes)

            logger.info('Target classes {} weight {}'.format([idx2class[c] for c in target_classes], target_weight))

            sample_weights = [0] * len(dataset)

            for index, (input, label) in enumerate(dataset):
                if label in target_classes:
                    sample_weights[index] = target_weight / num_target_classes
                else:
                    sample_weights[index] = (1.0 - target_weight) / (num_classes - num_target_classes)

        super().__init__(weights=sample_weights, num_samples=len(sample_weights))

def get_sampler(dataset, class_name, *args, **kwargs):
    if class_name != 'TargetClassSampler':
        logger.info('No sampler called `{}` is registered.'.format(class_name))
        return None
    
    return TargetClassSampler(dataset, **kwargs)
