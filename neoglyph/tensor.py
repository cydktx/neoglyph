import numpy as np


class Tensor:
    def __init__(self, data, shape=None):
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            arr = np.array(data, dtype=np.float32)
            if shape is not None:
                arr = arr.reshape(shape)
            self.data = arr
        self.shape = self.data.shape
        self.grad = None

    def detach(self):
        pass

    def __repr__(self):
        return f"Tensor({self.data.tolist()}, shape={list(self.shape)})"

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        return Tensor(self.data + other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        return Tensor(self.data - other)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        return Tensor(self.data * other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        return Tensor(self.data / other)
