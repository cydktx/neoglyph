import unittest
import numpy as np
from neoglyph.vm import NeoGlyphVM


class TestGradients(unittest.TestCase):
    def test_add_grad(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 2
        STORE a
        PUSH 3
        STORE b

        TAPE
        LOAD a
        LOAD b
        ADD
        STORE c
        UNTAPE

        GRAD

        LOAD a
        STORE ga
        LOAD b
        STORE gb
        HALT
        """
        vm.run(script)
        ga = vm.vars['ga']
        gb = vm.vars['gb']
        np.testing.assert_allclose(ga.grad, np.array([1.0]))
        np.testing.assert_allclose(gb.grad, np.array([1.0]))

    def test_mul_grad(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 2
        STORE a
        PUSH 3
        STORE b

        TAPE
        LOAD a
        LOAD b
        MUL
        STORE c
        UNTAPE

        GRAD

        LOAD a
        STORE ga
        LOAD b
        STORE gb
        HALT
        """
        vm.run(script)
        ga = vm.vars['ga']
        gb = vm.vars['gb']
        np.testing.assert_allclose(ga.grad, np.array([3.0]))
        np.testing.assert_allclose(gb.grad, np.array([2.0]))

    def test_chain_rule(self):
        vm = NeoGlyphVM()
        script = """
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

        GRAD

        LOAD a
        STORE ga
        LOAD b
        STORE gb
        HALT
        """
        vm.run(script)
        ga = vm.vars['ga']
        gb = vm.vars['gb']
        np.testing.assert_allclose(ga.grad, np.array([3.0]))
        np.testing.assert_allclose(gb.grad, np.array([8.0]))

    def test_relu_grad(self):
        vm = NeoGlyphVM()
        script = """
        PUSH -1 0 1 2
        STORE x

        TAPE
        LOAD x
        RELU
        STORE y
        UNTAPE

        GRAD

        LOAD x
        STORE gx
        HALT
        """
        vm.run(script)
        gx = vm.vars['gx']
        expected = np.array([0., 0., 1., 1.], dtype=np.float32)
        np.testing.assert_allclose(gx.grad, expected)

    def test_div_grad(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 10
        STORE a
        PUSH 2
        STORE b

        TAPE
        LOAD a
        LOAD b
        DIV
        STORE c
        UNTAPE

        GRAD

        LOAD a
        STORE ga
        LOAD b
        STORE gb
        HALT
        """
        vm.run(script)
        ga = vm.vars['ga']
        gb = vm.vars['gb']
        np.testing.assert_allclose(ga.grad, np.array([0.5]))
        np.testing.assert_allclose(gb.grad, np.array([-2.5]))

    def test_neg_grad(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 5
        STORE x

        TAPE
        LOAD x
        NEG
        STORE y
        UNTAPE

        GRAD

        LOAD x
        STORE gx
        HALT
        """
        vm.run(script)
        gx = vm.vars['gx']
        np.testing.assert_allclose(gx.grad, np.array([-1.0]))

    def test_matmul_grad(self):
        vm = NeoGlyphVM()
        script = """
        SHAPE (2,2)
        PUSH 1 2 3 4
        STORE A
        SHAPE (2,2)
        PUSH 5 6 7 8
        STORE B

        TAPE
        LOAD A
        LOAD B
        MATMUL
        STORE C
        UNTAPE

        SHAPE (2,2)
        PUSH 1 0 0 1
        GRAD

        LOAD A
        STORE gA
        LOAD B
        STORE gB
        HALT
        """
        vm.run(script)
        gA = vm.vars['gA']
        gB = vm.vars['gB']
        expected_gA = np.array([[5., 7.], [6., 8.]], dtype=np.float32)
        expected_gB = np.array([[1., 3.], [2., 4.]], dtype=np.float32)
        np.testing.assert_allclose(gA.grad, expected_gA)
        np.testing.assert_allclose(gB.grad, expected_gB)


if __name__ == '__main__':
    unittest.main()
