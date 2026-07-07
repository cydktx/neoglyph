import unittest
from neoglyph.vm import NeoGlyphVM


class TestOps(unittest.TestCase):
    def test_add_op(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 1 2 3
        PUSH 4 5 6
        ADD
        PRINT
        HALT
        """
        vm.run(script)

    def test_sub_op(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 10 20
        PUSH 3 5
        SUB
        PRINT
        HALT
        """
        vm.run(script)

    def test_mul_op(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 2 3
        PUSH 4 5
        MUL
        PRINT
        HALT
        """
        vm.run(script)

    def test_div_op(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 100 200
        PUSH 2 4
        DIV
        PRINT
        HALT
        """
        vm.run(script)

    def test_relu_op(self):
        vm = NeoGlyphVM()
        script = """
        PUSH -1 0 1 2
        RELU
        PRINT
        HALT
        """
        vm.run(script)

    def test_neg_op(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 1 -2 3
        NEG
        PRINT
        HALT
        """
        vm.run(script)

    def test_pow_op(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 2 3
        PUSH 3 2
        POW
        PRINT
        HALT
        """
        vm.run(script)

    def test_matmul_op(self):
        vm = NeoGlyphVM()
        script = """
        SHAPE (2,2)
        PUSH 1 2 3 4
        SHAPE (2,2)
        PUSH 5 6 7 8
        MATMUL
        PRINT
        HALT
        """
        vm.run(script)

    def test_store_load(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 42
        STORE x
        LOAD x
        PRINT
        HALT
        """
        vm.run(script)

    def test_jmp(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 1
        STORE count
        loop:
        LOAD count
        PRINT
        LOAD count
        PUSH 1
        ADD
        STORE count
        LOAD count
        PUSH 4
        SUB
        JMP_IF loop
        HALT
        """
        vm.run(script)


if __name__ == '__main__':
    unittest.main()
