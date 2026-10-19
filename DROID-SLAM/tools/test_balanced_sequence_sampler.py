"""CPU-only regression check for sequence-balanced DDP sampling."""

from collections import Counter
from pathlib import Path
import sys

from torch.utils.data import Dataset

# Keep the test executable as ``python tools/<script>.py`` from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "droid_slam"))

from data_readers.sampling import (
    BalancedSequenceDistributedSampler,
    sequence_sampling_weights,
)


class _UnevenScenes(Dataset):
    def __init__(self):
        # Scene 0 has 10 starts and scene 1 has 1,000 starts.
        self.dataset_index = [(0, index) for index in range(10)]
        self.dataset_index += [(1, index) for index in range(1000)]

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, index):
        return index


def main():
    dataset = _UnevenScenes()
    weights = sequence_sampling_weights(dataset)
    assert abs(float(weights[:10].sum()) - float(weights[10:].sum())) < 1e-10

    sampler = BalancedSequenceDistributedSampler(dataset, seed=7)
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(0)
    assert first == list(sampler), "sampler must be reproducible within an epoch"
    sampler.set_epoch(1)
    assert first != list(sampler), "sampler must change across epochs"

    counts = Counter(dataset.dataset_index[index][0] for index in first)
    ratio = counts[0] / max(counts[1], 1)
    assert 0.8 < ratio < 1.25, counts
    print("[PASS] balanced sequence sampler", dict(counts))


if __name__ == "__main__":
    main()
