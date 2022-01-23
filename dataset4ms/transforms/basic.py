import numpy as np
import mindspore._c_dataengine as cde

class Transform:
    def __init__(self, node) -> None:
        self.execute = cde.Execute(node)

    def __call__(self, *inputs):
        tensor_row = []
        for input in inputs:
            try:
                tensor_row.append(cde.Tensor(np.asarray(input)))
            except RuntimeError:
                raise TypeError("Invalid user input. Got {}: {}, cannot be converted into tensor." \
                                .format(type(input), input))
        outputs = self.execute(tensor_row)
        if len(inputs) == 1:
            return outputs[0].as_array()
        return [i.as_array() for i in outputs]

class OneHot(Transform):
    def __init__(self, num_classes):
        node = cde.OneHotOperation(num_classes)
        super().__init__(node)
