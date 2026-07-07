import unittest
import numpy as np
from neoglyph.tensor import Tensor


class TestTensor(unittest.TestCase):
    def test_init_from_list(self):
        t = Tensor([1, 2, 3])
        self.assertEqual(t.shape, (3,))
        np.testing.assert_array_equal(t.data, np.array([1., 2., 3.], dtype=np.float32))

    def test_init_from_numpy(self):
        arr = np.array([[1, 2], [3, 4]])
        t = Tensor(arr)
        self.assertEqual(t.shape, (2, 2))
        np.testing.assert_array_equal(t.data, arr.astype(np.float32))

    def test_init_with_shape(self):
        t = Tensor([1, 2, 3, 4], shape=(2, 2))
        self.assertEqual(t.shape, (2, 2))
        expected = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
        np.testing.assert_array_equal(t.data, expected)

    def test_grad_init(self):
        t = Tensor([1, 2, 3])
        self.assertIsNone(t.grad)

    def test_repr(self):
        t = Tensor([1, 2])
        repr_str = repr(t)
        self.assertIn('Tensor', repr_str)
        self.assertIn('[1.0, 2.0]', repr_str)
        self.assertIn('shape=[2]', repr_str)

    def test_add(self):
        a = Tensor([1, 2])
        b = Tensor([3, 4])
        c = a + b
        np.testing.assert_array_equal(c.data, np.array([4., 6.], dtype=np.float32))

    def test_sub(self):
        a = Tensor([5, 6])
        b = Tensor([2, 1])
        c = a - b
        np.testing.assert_array_equal(c.data, np.array([3., 5.], dtype=np.float32))

    def test_mul(self):
        a = Tensor([2, 3])
        b = Tensor([4, 5])
        c = a * b
        np.testing.assert_array_equal(c.data, np.array([8., 15.], dtype=np.float32))

    def test_div(self):
        a = Tensor([8, 15])
        b = Tensor([2, 5])
        c = a / b
        np.testing.assert_array_equal(c.data, np.array([4., 3.], dtype=np.float32))

    def test_add_scalar(self):
        a = Tensor([1, 2])
        c = a + 3
        np.testing.assert_array_equal(c.data, np.array([4., 5.], dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
