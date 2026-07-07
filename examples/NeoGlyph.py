"""
NeoGlyph
"""

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
        pass  # 双栈模式下无需操作

    def __repr__(self):
        return f"Tensor({self.data.tolist()}, shape={list(self.shape)})"


class NeoGlyphVM:
    def __init__(self):
        self.stack = []
        self.vars = {}
        self.labels = {}
        self.pc = 0
        self._current_shape = None
        self.recording = False          # TAPE 状态
        self.tape_ops = []              # 记录的反向操作序列
        self._obj_map = {}
        
    def load(self, code):
        """解析脚本，记录标签位置"""
        self.code = []
        for line in code.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            # 切分指令和参数
            parts = line.split()
            if parts[0].endswith(':'):
                self.labels[parts[0][:-1]] = len(self.code)
                continue
            self.code.append(parts)

    def run(self, code):
        self.load(code)
        self.pc = 0
        while self.pc < len(self.code):
            parts = self.code[self.pc]
            op = parts[0].upper()
            if op == 'PUSH':
                raw = ' '.join(parts[1:])
                vals = [float(x) for x in raw.replace(',', ' ').split()]
                shape = getattr(self, '_current_shape', None)
                if shape:
                    t = Tensor(vals, shape)
                    self._current_shape = None
                else:
                    t = Tensor(vals)
                self.stack.append(t)
            elif op == 'ADD':
                b = self.stack.pop()
                a = self.stack.pop()
                out = Tensor(a.data + b.data)
                self.stack.append(out)
                if self.recording:
                    out_id = id(out)
                    self.tape_ops.append(('add', out_id, id(a), id(b)))
                    self._obj_map[out_id] = out
                    self._obj_map[id(a)] = a
                    self._obj_map[id(b)] = b

            elif op == 'SUB':
                b = self.stack.pop()
                a = self.stack.pop()
                out = Tensor(a.data - b.data)
                self.stack.append(out)
                if self.recording:
                    out_id = id(out)
                    self.tape_ops.append(('sub', out_id, id(a), id(b)))
                    self._obj_map[out_id] = out
                    self._obj_map[id(a)] = a
                    self._obj_map[id(b)] = b

            elif op == 'MUL':
                b = self.stack.pop()
                a = self.stack.pop()
                out = Tensor(a.data * b.data)
                self.stack.append(out)
                if self.recording:
                    out_id = id(out)
                    self.tape_ops.append(('mul', out_id, id(a), id(b),
                                          a.data.copy(), b.data.copy()))
                    self._obj_map[out_id] = out
                    self._obj_map[id(a)] = a
                    self._obj_map[id(b)] = b
            elif op == 'DIV':
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(Tensor(a.data / b.data))
            elif op == 'SHAPE':
                shape_str = ' '.join(parts[1:]).strip('()')
                shape = tuple(int(x) for x in shape_str.split(','))
                self._current_shape = shape
            elif op == 'POP':
                self.stack.pop()
            elif op == 'STORE':
                var = parts[1]
                self.vars[var] = self.stack.pop()
            elif op == 'LOAD':
                var = parts[1]
                self.stack.append(self.vars[var])
            elif op == 'PRINT':
                val = self.stack.pop()
                if isinstance(val, Tensor) and val.grad is not None:
                    print(Tensor(val.grad))
                else:
                    print(val)
            elif op == 'TAPE':
                self.recording = True
                self.tape_ops = []

            elif op == 'UNTAPE':
                self.recording = False

            elif op == 'GRAD':
                # 从栈顶弹出梯度种子（VJP），若栈为空则默认 1.0
                if self.stack:
                    seed = self.stack.pop()
                    grad_val = seed.data if isinstance(seed, Tensor) else seed
                else:
                    grad_val = np.array([1.0], dtype=np.float32)

                # 初始化所有参与节点的梯度缓存
                grad_table = {}
                # 获取 tape 中最后一条记录的输出节点 ID 作为初始种子
                if self.tape_ops:
                    last_out_id = self.tape_ops[-1][1]
                    grad_table[last_out_id] = np.array(grad_val, dtype=np.float32)

                # 逆序遍历反向操作
                for op_record in reversed(self.tape_ops):
                    op_type, out_id, *args = op_record
                    grad_out = grad_table.get(out_id, None)

                    if grad_out is None:
                        continue

                    if op_type == 'add':
                        a_id, b_id = args
                        if a_id not in grad_table:
                            grad_table[a_id] = np.zeros_like(self._obj_map[a_id].data)
                        if b_id not in grad_table:
                            grad_table[b_id] = np.zeros_like(self._obj_map[b_id].data)
                        grad_table[a_id] += grad_out
                        grad_table[b_id] += grad_out
                    elif op_type == 'sub':
                        a_id, b_id = args
                        if a_id not in grad_table:
                            grad_table[a_id] = np.zeros_like(self._obj_map[a_id].data)
                        if b_id not in grad_table:
                            grad_table[b_id] = np.zeros_like(self._obj_map[b_id].data)
                        grad_table[a_id] += grad_out
                        grad_table[b_id] -= grad_out
                    elif op_type == 'mul':
                        a_id, b_id, a_data, b_data = args
                        if a_id not in grad_table:
                            grad_table[a_id] = np.zeros_like(self._obj_map[a_id].data)
                        if b_id not in grad_table:
                            grad_table[b_id] = np.zeros_like(self._obj_map[b_id].data)
                        grad_table[a_id] += b_data * grad_out
                        grad_table[b_id] += a_data * grad_out
                    # 后续添加 relu, matmul 等分支

                # 将梯度写回各张量
                for obj_id, grad_arr in grad_table.items():
                    # 通过 id 找回 Tensor 对象（需要一个映射表）
                    if obj_id in self._obj_map:
                        tensor = self._obj_map[obj_id]
                        if tensor.grad is None:
                            tensor.grad = np.zeros_like(tensor.data)
                        tensor.grad += grad_arr

            elif op == 'HALT':
                break
            elif op == 'JMP':
                self.pc = self.labels[parts[1]]
                continue
            elif op == 'JMP_IF':
                cond = self.stack.pop()
                if not cond:
                    self.pc = self.labels[parts[1]]
                    continue
            else:
                raise ValueError(f"未知指令: {op}")
            self.pc += 1

if __name__ == "__main__":
    vm = NeoGlyphVM()
    script = """
    // 计算 d = (a+b)*b，验证梯度
    PUSH 2
    STORE a
    PUSH 3
    STORE b

    TAPE
    LOAD a
    LOAD b
    ADD
    LOAD b
    MUL
    STORE d
    UNTAPE

    LOAD d
    GRAD

    LOAD a
    PRINT
    LOAD b
    PRINT
    HALT
    """
    vm.run(script)