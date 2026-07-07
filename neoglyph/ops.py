import numpy as np
from .tensor import Tensor

_op_registry = {}


def register_op(name):
    def decorator(cls):
        _op_registry[name.upper()] = cls
        return cls
    return decorator


def get_op(name):
    return _op_registry.get(name.upper())


class OpBase:
    name = None

    def execute(self, vm):
        raise NotImplementedError

    def record(self, vm, out, *inputs):
        if vm.recording:
            out_id = id(out)
            args = [id(inp) for inp in inputs]
            extra = self._get_extra_data(*inputs)
            vm.tape_ops.append((self.name, out_id, *args, *extra))
            vm._obj_map[out_id] = out
            for inp in inputs:
                vm._obj_map[id(inp)] = inp

    def _get_extra_data(self, *inputs):
        return []

    def backward(self, vm, out_id, grad_out, *args):
        raise NotImplementedError


@register_op('ADD')
class AddOp(OpBase):
    name = 'add'

    def execute(self, vm):
        b = vm.stack.pop()
        a = vm.stack.pop()
        out = Tensor(a.data + b.data)
        vm.stack.append(out)
        self.record(vm, out, a, b)

    def backward(self, vm, out_id, grad_out, a_id, b_id):
        a = vm._obj_map[a_id]
        b = vm._obj_map[b_id]
        return {a_id: grad_out, b_id: grad_out}


@register_op('SUB')
class SubOp(OpBase):
    name = 'sub'

    def execute(self, vm):
        b = vm.stack.pop()
        a = vm.stack.pop()
        out = Tensor(a.data - b.data)
        vm.stack.append(out)
        self.record(vm, out, a, b)

    def backward(self, vm, out_id, grad_out, a_id, b_id):
        a = vm._obj_map[a_id]
        b = vm._obj_map[b_id]
        return {a_id: grad_out, b_id: -grad_out}


@register_op('MUL')
class MulOp(OpBase):
    name = 'mul'

    def execute(self, vm):
        b = vm.stack.pop()
        a = vm.stack.pop()
        out = Tensor(a.data * b.data)
        vm.stack.append(out)
        self.record(vm, out, a, b)

    def _get_extra_data(self, a, b):
        return [a.data.copy(), b.data.copy()]

    def backward(self, vm, out_id, grad_out, a_id, b_id, a_data, b_data):
        return {a_id: b_data * grad_out, b_id: a_data * grad_out}


@register_op('DIV')
class DivOp(OpBase):
    name = 'div'

    def execute(self, vm):
        b = vm.stack.pop()
        a = vm.stack.pop()
        safe_b = np.where(np.abs(b.data) < 1e-8, 1e-8, b.data)
        out = Tensor(a.data / safe_b)
        out.data = np.where(np.isfinite(out.data), out.data, 0.0)
        vm.stack.append(out)
        self.record(vm, out, a, b)

    def _get_extra_data(self, a, b):
        return [a.data.copy(), b.data.copy()]

    def backward(self, vm, out_id, grad_out, a_id, b_id, a_data, b_data):
        safe_b = np.where(np.abs(b_data) < 1e-8, 1e-8, b_data)
        grad_a = grad_out / safe_b
        grad_b = -a_data * grad_out / (safe_b ** 2)
        grad_a = np.where(np.isfinite(grad_a), grad_a, 0.0)
        grad_b = np.where(np.isfinite(grad_b), grad_b, 0.0)
        return {a_id: grad_a, b_id: grad_b}


@register_op('RELU')
class ReluOp(OpBase):
    name = 'relu'

    def execute(self, vm):
        a = vm.stack.pop()
        out = Tensor(np.maximum(0, a.data))
        vm.stack.append(out)
        self.record(vm, out, a)

    def _get_extra_data(self, a):
        return [a.data.copy()]

    def backward(self, vm, out_id, grad_out, a_id, a_data):
        mask = (a_data > 0).astype(np.float32)
        return {a_id: mask * grad_out}


@register_op('NEG')
class NegOp(OpBase):
    name = 'neg'

    def execute(self, vm):
        a = vm.stack.pop()
        out = Tensor(-a.data)
        vm.stack.append(out)
        self.record(vm, out, a)

    def backward(self, vm, out_id, grad_out, a_id):
        return {a_id: -grad_out}


@register_op('POW')
class PowOp(OpBase):
    name = 'pow'

    def execute(self, vm):
        b = vm.stack.pop()
        a = vm.stack.pop()
        safe_a = np.where(np.abs(a.data) < 1e-8, 1e-8, a.data)
        out = Tensor(safe_a ** b.data)
        out.data = np.where(np.isfinite(out.data), out.data, 0.0)
        vm.stack.append(out)
        self.record(vm, out, a, b)

    def _get_extra_data(self, a, b):
        return [a.data.copy(), b.data.copy()]

    def backward(self, vm, out_id, grad_out, a_id, b_id, a_data, b_data):
        safe_a = np.where(np.abs(a_data) < 1e-8, 1e-8, a_data)
        grad_a = b_data * (safe_a ** (b_data - 1)) * grad_out
        grad_b = (safe_a ** b_data) * np.log(safe_a + 1e-10) * grad_out
        grad_a = np.where(np.isfinite(grad_a), grad_a, 0.0)
        grad_b = np.where(np.isfinite(grad_b), grad_b, 0.0)
        return {a_id: grad_a, b_id: grad_b}


@register_op('MATMUL')
class MatMulOp(OpBase):
    name = 'matmul'

    def execute(self, vm):
        b = vm.stack.pop()
        a = vm.stack.pop()
        out = Tensor(np.dot(a.data, b.data))
        vm.stack.append(out)
        self.record(vm, out, a, b)

    def _get_extra_data(self, a, b):
        return [a.data.copy(), b.data.copy()]

    def backward(self, vm, out_id, grad_out, a_id, b_id, a_data, b_data):
        return {
            a_id: np.dot(grad_out, b_data.T),
            b_id: np.dot(a_data.T, grad_out)
        }


@register_op('SIN')
class SinOp(OpBase):
    name = 'sin'

    def execute(self, vm):
        a = vm.stack.pop()
        out = Tensor(np.sin(a.data))
        vm.stack.append(out)
        self.record(vm, out, a)

    def backward(self, vm, out_id, grad_out, a_id):
        a = vm._obj_map[a_id]
        return {a_id: np.cos(a.data) * grad_out}


@register_op('COS')
class CosOp(OpBase):
    name = 'cos'

    def execute(self, vm):
        a = vm.stack.pop()
        out = Tensor(np.cos(a.data))
        vm.stack.append(out)
        self.record(vm, out, a)

    def backward(self, vm, out_id, grad_out, a_id):
        a = vm._obj_map[a_id]
        return {a_id: -np.sin(a.data) * grad_out}


@register_op('EXP')
class ExpOp(OpBase):
    name = 'exp'

    def execute(self, vm):
        a = vm.stack.pop()
        clipped = np.clip(a.data, -80, 80)  # 防止溢出
        out = Tensor(np.exp(clipped))
        vm.stack.append(out)
        self.record(vm, out, a)

    def backward(self, vm, out_id, grad_out, a_id):
        out = vm._obj_map[out_id]
        return {a_id: out.data * grad_out}


@register_op('LOG')
class LogOp(OpBase):
    name = 'log'

    def execute(self, vm):
        a = vm.stack.pop()
        safe_a = np.maximum(a.data, 1e-10)
        out = Tensor(np.log(safe_a))
        vm.stack.append(out)
        self.record(vm, out, a)

    def backward(self, vm, out_id, grad_out, a_id):
        a = vm._obj_map[a_id]
        safe_a = np.maximum(a.data, 1e-10)
        return {a_id: grad_out / safe_a}
