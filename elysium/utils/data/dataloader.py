import os
from elysium import np
from concurrent.futures import ThreadPoolExecutor, as_completed
class Sampler:
    def __init__(self, data_source):
        """Base class for all samplers."""
        self.data_source = data_source
    def __iter__(self):raise NotImplementedError
    def __len__(self):return len(self.data_source)

class RandomSampler(Sampler):
    def __iter__(self):return iter(np.random.permutation(len(self.data_source)))

class SequentialSampler(Sampler):
    def __iter__(self):return iter(range(len(self.data_source)))

class BatchSampler:
    def __init__(self, sampler, batch_size, drop_last=False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
    def __iter__(self):
        """Yield batches of indices according to the sampling strategy."""
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
    def __len__(self):
        """Return the number of batches."""
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

def default_collate_fn(batch):
    """Default collate function that stacks data into a batch."""
    batch_data, batch_labels = zip(*batch)
    return batch_data, batch_labels

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=1, sampler=None, batch_sampler=None, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.sampler = sampler or (RandomSampler(dataset) if shuffle else SequentialSampler(dataset))
        self.batch_sampler = batch_sampler or BatchSampler(self.sampler, batch_size)
        self.collate_fn = collate_fn or default_collate_fn
    def __iter__(self):
        """Create an iterator that returns batches of data."""
        self.batch_sampler_iter = iter(self.batch_sampler)
        return self
    def __next__(self):
        """Return the next batch of data."""
        batch_indices = next(self.batch_sampler_iter)  # Get the next batch of indices
        if self.num_workers > 1:
            # Load data in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self.dataset.__getitem__, idx) for idx in batch_indices]
                batch = [f.result() for f in as_completed(futures)]
        else:
            # Load data sequentially
            batch = [self.dataset[i] for i in batch_indices]
        return self.collate_fn(batch)
    def __len__(self):
        """Return the number of batches per epoch."""
        return len(self.batch_sampler)
