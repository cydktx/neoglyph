"""
用 NeoGlyphVM 训练微型神经网络 — XOR 问题
网络结构: 2 → 4 → 1 (单隐藏层)
"""
import numpy as np
from neoglyph import Tensor, NeoGlyphVM
from neoglyph.ops import get_op


def gen_script(w1, b1, w2, b2, x1, x2):
    w1s = ' '.join(f'{v:.6f}' for row in w1 for v in row)
    b1s = ' '.join(f'{v:.6f}' for v in b1.flat)
    w2s = ' '.join(f'{v:.6f}' for row in w2 for v in row)
    return f"""
    SHAPE (1,2)
    PUSH {x1} {x2}
    STORE X
    SHAPE (2,4)
    PUSH {w1s}
    STORE W1
    SHAPE (1,4)
    PUSH {b1s}
    STORE B1
    SHAPE (4,1)
    PUSH {w2s}
    STORE W2
    SHAPE (1,1)
    PUSH {b2.flat[0]:.6f}
    STORE B2
    TAPE
    LOAD X
    LOAD W1
    MATMUL
    LOAD B1
    ADD
    RELU
    STORE H
    LOAD H
    LOAD W2
    MATMUL
    LOAD B2
    ADD
    STORE out
    UNTAPE
    HALT
    """


def forward(w1, b1, w2, b2, x1, x2):
    vm = NeoGlyphVM(verbose=False)
    vm.run(gen_script(w1, b1, w2, b2, x1, x2))
    return vm


def train_xor(steps=3000, lr=0.5):
    np.random.seed(42)
    w1 = np.random.randn(2, 4) * 0.5
    b1 = np.zeros((1, 4))
    w2 = np.random.randn(4, 1) * 0.5
    b2 = np.zeros((1, 1))

    data = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

    for step in range(steps):
        total_loss = 0.0
        gw1 = np.zeros_like(w1)
        gb1 = np.zeros_like(b1)
        gw2 = np.zeros_like(w2)
        gb2 = np.zeros_like(b2)

        for x1, x2, y_true in data:
            vm = forward(w1, b1, w2, b2, x1, x2)
            pred = vm.vars['out'].data[0, 0]
            loss = 0.5 * (pred - y_true) ** 2
            total_loss += loss
            dout = pred - y_true

            tape = vm.tape_ops
            obj_map = vm._obj_map
            grad_table = {}
            out_id = tape[-1][1]
            grad_table[out_id] = np.array([[dout]], dtype=np.float32)

            for op_record in reversed(tape):
                oid = op_record[1]
                if oid not in grad_table:
                    continue
                args = op_record[2:]
                op_class = get_op(op_record[0])
                if op_class:
                    grads = op_class().backward(vm, oid, grad_table[oid], *args)
                    for oid2, g in grads.items():
                        grad_table[oid2] = grad_table.get(oid2, np.zeros_like(obj_map[oid2].data)) + g

            for name, t in vm.vars.items():
                tid = id(t)
                if tid in grad_table:
                    g = grad_table[tid].reshape(t.data.shape)
                    if name == 'W1':
                        gw1 += g
                    elif name == 'B1':
                        gb1 += g
                    elif name == 'W2':
                        gw2 += g
                    elif name == 'B2':
                        gb2 += g

        w1 -= lr * gw1 / 4
        b1 -= lr * gb1 / 4
        w2 -= lr * gw2 / 4
        b2 -= lr * gb2 / 4

        if step % 500 == 0:
            print(f"Step {step:4d}  Loss: {total_loss/4:.6f}")

    print("\n=== 训练结果 ===")
    for x1, x2, y_true in data:
        vm = forward(w1, b1, w2, b2, x1, x2)
        out = vm.vars['out'].data[0, 0]
        print(f"X({x1},{x2}) → {out:.4f}  (期望: {y_true})  ✅" if abs(out - y_true) < 0.5 else f"X({x1},{x2}) → {out:.4f}  (期望: {y_true})")


if __name__ == "__main__":
    train_xor(steps=3000, lr=0.5)
