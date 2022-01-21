from signal import raise_signal
import mindspore._c_dataengine as cde

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
        self.source_len = kwargs.get('source_len', None)

        self.ir_tree = []
        self._init_generator_node()

    def _init_cde_runtime_context(self):
        self._runtime_context = cde.PythonRuntimeContext()
        self._runtime_context.Init()
    
    def _init_generator_node(self):
        node = cde.GeneratorNode(
            self.dataset,
            self.column_names,
            self.column_types,
            self.source_len,
            self.sampler,
            self.num_parallel_workers
        )
        self.ir_tree.append(node)

    def _transform(self, data):
        pass

    def map(self, fn):
        pass

    def __iter__(self):
        pass

    def _get_iterator(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.dataset)