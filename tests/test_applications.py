import unittest
import numpy as np
from neoglyph.applications import SymbolicRegressor, PhysicsDiscoverer


class TestSymbolicRegressor(unittest.TestCase):
    def test_init(self):
        reg = SymbolicRegressor(pop_size=10, max_depth=2, generations=5)
        self.assertEqual(reg.pop_size, 10)
        self.assertEqual(reg.max_depth, 2)
        self.assertEqual(reg.generations, 5)

    def test_fit_predict_linear(self):
        x = np.linspace(-3, 3, 15)
        y = 2 * x + 1

        reg = SymbolicRegressor(
            pop_size=40,
            max_depth=2,
            generations=50,
            random_state=42,
        )
        reg.fit(x, y)

        self.assertIsNotNone(reg.best_genome_)
        r2 = reg.score(x, y)
        self.assertGreater(r2, 0.8)

    def test_predict_returns_array(self):
        x = np.linspace(-3, 3, 10)
        y = 2 * x + 1

        reg = SymbolicRegressor(
            pop_size=30,
            max_depth=2,
            generations=30,
            random_state=42,
        )
        reg.fit(x, y)

        pred = reg.predict(x)
        self.assertEqual(len(pred), len(x))
        self.assertEqual(pred.shape, x.shape)

    def test_best_expression_before_fit(self):
        reg = SymbolicRegressor()
        self.assertEqual(reg.best_expression(), "N/A")

    def test_predict_before_fit_raises(self):
        reg = SymbolicRegressor()
        with self.assertRaises(ValueError):
            reg.predict([1, 2, 3])

    def test_history_recorded(self):
        x = np.linspace(-3, 3, 10)
        y = 2 * x + 1

        reg = SymbolicRegressor(
            pop_size=20,
            max_depth=2,
            generations=10,
            random_state=42,
        )
        reg.fit(x, y)

        self.assertEqual(len(reg.history_), 10)
        self.assertIn('best_fitness', reg.history_[0])
        self.assertIn('generation', reg.history_[0])
        self.assertIn('avg_fitness', reg.history_[0])

    def test_fit_with_2d_input(self):
        x = np.linspace(-3, 3, 10).reshape(-1, 1)
        y = 2 * x.ravel() + 1

        reg = SymbolicRegressor(
            pop_size=20,
            max_depth=2,
            generations=20,
            random_state=42,
        )
        reg.fit(x, y)
        pred = reg.predict(x)
        self.assertEqual(len(pred), len(x))


class TestPhysicsDiscoverer(unittest.TestCase):
    def test_init(self):
        disc = PhysicsDiscoverer(pop_size=20, max_depth=3, generations=10)
        self.assertIsNotNone(disc.regressor)

    def test_discover_linear(self):
        x = np.linspace(0.1, 1.0, 10)
        y = 10.0 * x

        disc = PhysicsDiscoverer(
            pop_size=30,
            max_depth=2,
            generations=40,
            random_state=42,
        )
        result = disc.discover(x, y)

        self.assertIn('expression', result)
        self.assertIn('r2_score', result)
        self.assertIn('mse', result)
        self.assertIn('mae', result)
        self.assertIn('complexity', result)
        self.assertIn('genome', result)
        self.assertIn('history', result)

        self.assertGreater(result['r2_score'], 0.7)

    def test_discover_quadratic(self):
        x = np.linspace(0.5, 5.0, 12)
        y = 4.9 * x ** 2

        disc = PhysicsDiscoverer(
            pop_size=60,
            max_depth=3,
            generations=80,
            random_state=42,
        )
        result = disc.discover(x, y)

        self.assertGreater(result['r2_score'], 0.75)


class TestMultiVariable(unittest.TestCase):
    def test_multi_variable_fit_predict(self):
        """f(x, y) = 2x + 3y"""
        X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], dtype=np.float64)
        y = 2 * X[:, 0] + 3 * X[:, 1]

        reg = SymbolicRegressor(
            pop_size=40,
            max_depth=2,
            generations=50,
            random_state=42,
        )
        reg.fit(X, y)

        self.assertIsNotNone(reg.best_genome_)
        r2 = reg.score(X, y)
        self.assertGreater(r2, 0.7)

    def test_multi_variable_predict_shape(self):
        X = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float64)
        y = np.array([5, 10, 15], dtype=np.float64)

        reg = SymbolicRegressor(
            pop_size=20,
            max_depth=2,
            generations=20,
            random_state=42,
        )
        reg.fit(X, y)
        pred = reg.predict(X)
        self.assertEqual(len(pred), len(X))

    def test_tree_multi_variable_vm_code(self):
        """多变量表达式树应生成正确的 VM 代码"""
        from neoglyph.genome import TreeGenome, OperationNode, VariableNode, ConstantNode
        # f(x, y) = x * y
        tree = TreeGenome(OperationNode('MUL',
            VariableNode('x'), VariableNode('y')))
        code = tree.to_vm_code()
        self.assertIn('LOAD', code)
        self.assertIn('MUL', code)

    def test_tree_multi_variable_evaluate_array(self):
        """多变量 evaluate_array 应正确计算"""
        from neoglyph.genome import TreeGenome, OperationNode, VariableNode, ConstantNode
        # f(x, y) = x + y
        tree = TreeGenome(OperationNode('ADD',
            VariableNode('x'), VariableNode('y')))
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        y = X[:, 0] + X[:, 1]
        result = tree.evaluate_array(X, y)
        self.assertAlmostEqual(result['accuracy'], 1.0, places=2)
        self.assertLess(result['mse'], 0.001)


if __name__ == '__main__':
    unittest.main()
