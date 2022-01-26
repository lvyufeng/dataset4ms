import logging
import numpy as np
from typing import OrderedDict
import mindspore._c_dataengine as cde

OffloadToManualOffloadMode = {
    None: cde.ManualOffloadMode.UNSPECIFIED,
    False: cde.ManualOffloadMode.DISABLED,
    True: cde.ManualOffloadMode.ENABLED
}

class Dataset:
    def __init__(self) -> None:
        pass

    def __getitem__(self, index):
        raise NotImplementedError(f"__getitem__ not implement in class {self.__class__.__name__}")
    
    def __len__(self):
        raise NotImplementedError(f"__len__ not implement in class {self.__class__.__name__}")


class DataProcess:
    def __init__(self, dataset, num_epochs=1, sampler=None, num_parallel_workers=1, **kwargs) -> None:
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.sampler = sampler

        # init cde runtime context
        self._init_cde_runtime_context()

        self.num_parallel_workers = num_parallel_workers
        self.column_names = kwargs.get('column_names', None)
        self.column_types = kwargs.get('column_types', None)
        self.source_len = len(self.dataset)

        self.shuffle = kwargs.get('shuffle', False)
        self.shuffle_buffer = kwargs.get('shuffle_buffer', 32)


        if self.column_names is None:
            self.column_names = [str(idx) for idx in range(len(self.dataset[0]))]

        self.ir_tree = []
        self._init_generator_node()

        self._transforms = OrderedDict({name: [] for name in self.column_names})

    def _init_cde_runtime_context(self):
        self._runtime_context = cde.PythonRuntimeContext()
        self._runtime_context.Init()
    
    def _init_generator_node(self):
        node = cde.GeneratorNode(
            (lambda: _iter_fn(self.dataset)),
            self.column_names,
            [],
            self.source_len,
            None,
            self.num_parallel_workers
        )
        self.ir_tree.append(node)

    def _transform(self, data):
        def transform_column(column_data, transforms):
            for transform in transforms:
                column_data = transform(column_data)
            return column_data

        return list(map(transform_column, data, self._transforms.values()))

    def map(self, fn, column_name=None):
        if column_name is None:
            column_name = self.column_names[0]
            logging.warning(f"The column name is None, the transform {fn} will be applied "
                            f"on the first column with automatic named {column_name}")
        self._transforms[column_name].append(fn)
        self.ir_tree.append(
            cde.MapNode(self.ir_tree[-1], [fn], [column_name], [], [], [], 16, OffloadToManualOffloadMode[None])
        )

    def __iter__(self):
        consumer = cde.PythonIteratorConsumer(self.num_epochs)
        consumer.Init(self.ir_tree[0])
        self._runtime_context.AssignConsumer(consumer)
        self._iterator = self._runtime_context.GetConsumer()
        for _ in range(self.source_len):
            yield self._iterator.GetNextAsList()

    def _get_iterator(self):
        pass

    def __getitem__(self, index):
        if self._transforms:
            return self._transform(self.dataset[index])
        return self.dataset[index]

    def __len__(self):
        return self.source_len


def _iter_fn(dataset):
    for val in dataset:
        yield val