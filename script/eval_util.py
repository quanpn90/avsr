import torch
from torch.utils.data import DataLoader, Subset
from datasets import Dataset, IterableDataset
from transformers.feature_extraction_utils import BatchFeature



def split_dataset(dataset, num_chunks):
    """Splits dataset into `num_chunks` evenly."""
    num_samples = len(dataset)
    indices = torch.arange(num_samples)
    chunk_size = (num_samples + num_chunks - 1) // num_chunks
    return [Subset(dataset, indices[i * chunk_size: (i + 1) * chunk_size]) for i in range(num_chunks)]


def split_iterable_dataset(iterable_dataset, num_chunks):
    data = []
    for sample in iterable_dataset:
        data.append(sample)

    num_samples = len(data)
    chunk_size = (num_samples + num_chunks - 1) // num_chunks

    return [
        IterableDataset.from_generator(lambda chunk=data[i * chunk_size: (i + 1) * chunk_size]: iter(chunk))
        for i in range(num_chunks)
    ]


def to_cuda_recursive(data, device, dtype=torch.float32):
    if isinstance(data, torch.Tensor):
        return data.to(device, dtype=dtype)
    elif isinstance(data, list):
        return [to_cuda_recursive(item, device, dtype=dtype) for item in data]
    elif isinstance(data, dict):
        return {key: to_cuda_recursive(value, device, dtype=dtype) for key, value in data.items()}
    elif isinstance(data, BatchFeature):
        return {key: to_cuda_recursive(value, device, dtype=dtype) for key, value in data.items()}
    else:
        return data

