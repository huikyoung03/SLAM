"""Sampling helpers for ordered multi-sequence visual-inertial datasets."""

from collections import Counter
import math

import torch
from torch.utils.data import ConcatDataset, Sampler


def _local_sequence_ids(dataset, dataset_prefix):
    """Return one stable sequence identifier per dataset sample when available."""

    dataset_index = getattr(dataset, "dataset_index", None)
    if dataset_index is None or len(dataset_index) != len(dataset):
        return [(dataset_prefix, 0)] * len(dataset)

    sequence_ids = []
    for sample in dataset_index:
        # Ordered EuRoC/TartanAir readers store (scene_id, start_frame).
        scene_id = sample[0] if isinstance(sample, tuple) else 0
        sequence_ids.append((dataset_prefix, int(scene_id)))
    return sequence_ids


def sequence_sampling_weights(dataset):
    """Give every discovered sequence equal total sampling probability.

    The default random sampler selects clip starts uniformly. Consequently, a
    long recording contributes far more updates than a short recording. This
    helper preserves random clip selection while assigning every sequence the
    same *expected* number of updates in an epoch.
    """

    if isinstance(dataset, ConcatDataset):
        sequence_ids = []
        for dataset_id, child in enumerate(dataset.datasets):
            sequence_ids.extend(_local_sequence_ids(child, dataset_id))
    else:
        sequence_ids = _local_sequence_ids(dataset, 0)

    if not sequence_ids:
        raise ValueError("cannot construct a sampler for an empty dataset")

    counts = Counter(sequence_ids)
    return torch.as_tensor(
        [1.0 / counts[sequence_id] for sequence_id in sequence_ids],
        dtype=torch.double,
    )


class BalancedSequenceDistributedSampler(Sampler):
    """Deterministic DDP-compatible weighted sampler for multi-sequence clips."""

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        seed=0,
        drop_last=False,
    ):
        if num_replicas is None:
            num_replicas = 1
        if rank is None:
            rank = 0
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed sampler rank/replica count")

        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.weights = sequence_sampling_weights(dataset)

        if self.drop_last and len(dataset) % self.num_replicas != 0:
            self.num_samples = math.floor(len(dataset) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        ).tolist()
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
