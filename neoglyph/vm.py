import numpy as np
from .tensor import Tensor
from .ops import get_op
from .profiler import Profiler


class NeoGlyphVM:
    def __init__(self, verbose=False):
        self.stack = []
        self.vars = {}
        self.labels = {}
        self.pc = 0
        self._current_shape = None
        self.recording = False
        self.tape_ops = []
        self._obj_map = {}
        self.profiler = Profiler()
        self.verbose = verbose

    def load(self, code):
        self.code = []
        for line in code.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            parts = line.split()
            if parts[0].endswith(':'):
                self.labels[parts[0][:-1]] = len(self.code)
                continue
            self.code.append(parts)

    def run(self, code):
        self.profiler.reset()
        self.profiler.start()
        try:
            self.load(code)
            self.pc = 0
            while self.pc < len(self.code):
                parts = self.code[self.pc]
                op = parts[0].upper()

                self.profiler.on_instruction_start(op)

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
                    self.profiler.on_tensor_created(t)

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
                    if self.verbose:
                        if isinstance(val, Tensor) and val.grad is not None:
                            print(Tensor(val.grad))
                        else:
                            print(val)

                elif op == 'TAPE':
                    self.recording = True
                    self.tape_ops = []

                elif op == 'UNTAPE':
                    self.recording = False
                    self.profiler.on_tape_record(self.tape_ops)

                elif op == 'GRAD':
                    self._compute_grad()
                    self.profiler.on_tape_record(self.tape_ops)

                elif op == 'HALT':
                    self.profiler.on_instruction_end(op)
                    break

                elif op == 'JMP':
                    self.pc = self.labels[parts[1]]
                    self.profiler.on_instruction_end(op)
                    continue

                elif op == 'JMP_IF':
                    cond = self.stack.pop()
                    cond_val = cond.data[0] if hasattr(cond, 'data') else cond
                    if not cond_val:
                        self.pc = self.labels[parts[1]]
                        self.profiler.on_instruction_end(op)
                        continue
                    self.profiler.on_instruction_end(op)

                else:
                    op_class = get_op(op)
                    if op_class:
                        op_class().execute(self)
                        self.profiler.on_tensor_op()
                    else:
                        raise ValueError(f"未知指令: {op}")

                self.profiler.on_instruction_end(op)
                self.pc += 1
        except Exception as e:
            self.profiler.on_error()
            raise
        finally:
            self.profiler.stop(self._obj_map)

    def get_profile_report(self):
        return self.profiler.get_report()

    def get_fitness_metrics(self, **kwargs):
        return self.profiler.get_fitness_metrics(**kwargs)

    def _compute_grad(self):
        grad_table = {}
        
        if self.tape_ops:
            last_out_id = self.tape_ops[-1][1]
            last_tensor = None
            
            # 从栈顶往下找，匹配最后一个tape输出（用id()匹配）
            for item in reversed(self.stack):
                if isinstance(item, Tensor) and id(item) == last_out_id:
                    last_tensor = item
                    break
            
            if last_tensor is not None:
                grad_val = last_tensor.data
            else:
                # 找不到的话，回退到栈顶
                if self.stack:
                    seed = self.stack[-1]
                    grad_val = seed.data if isinstance(seed, Tensor) else seed
                else:
                    grad_val = np.array([1.0], dtype=np.float32)
            
            grad_table[last_out_id] = np.array(grad_val, dtype=np.float32)
            
            # 多分支支持：如果vars中有多个输出tensor，也加入梯度种子
            tape_out_ids = set(op[1] for op in self.tape_ops)
            for var_name, var_val in self.vars.items():
                if isinstance(var_val, Tensor) and id(var_val) in tape_out_ids:
                    var_id = id(var_val)
                    if var_id != last_out_id:
                        if var_id not in grad_table:
                            grad_table[var_id] = np.zeros_like(var_val.data)

        for op_record in reversed(self.tape_ops):
            op_type = op_record[0]
            out_id = op_record[1]
            args = op_record[2:]

            grad_out = grad_table.get(out_id, None)
            if grad_out is None:
                continue

            op_class = get_op(op_type)
            if op_class:
                grads = op_class().backward(self, out_id, grad_out, *args)
                for obj_id, grad_arr in grads.items():
                    if obj_id not in grad_table:
                        grad_table[obj_id] = np.zeros_like(self._obj_map[obj_id].data)
                    grad_table[obj_id] += grad_arr

        for obj_id, grad_arr in grad_table.items():
            if obj_id in self._obj_map:
                tensor = self._obj_map[obj_id]
                if tensor.grad is None:
                    tensor.grad = np.zeros_like(tensor.data)
                tensor.grad += grad_arr
