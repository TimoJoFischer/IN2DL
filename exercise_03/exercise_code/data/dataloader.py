"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):

        if self.shuffle:
            index_iterator = iter(np.random.permutation(len(self.dataset)))
        else:
            index_iterator = iter(len(self.dataset))

        batch=[]
        for index in index_iterator:
            batch.append(self.dataset[index]['data'])
            if len(batch)==self.batch_size:
                yield {'data':np.array(batch)}
                batch=[]
        if not self.drop_last:
            yield {'data':np.array(batch)}

    def __len__(self):
        if(self.drop_last):
            length = int(len(self.dataset) / self.batch_size)
        else:
            length = int(np.ceil((len(self.dataset) / self.batch_size)))

        return length
