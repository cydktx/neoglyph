import unittest
import numpy as np
from neoglyph import (
    TreeGenome, ConstantNode, VariableNode, OperationNode,
    NeoGlyphVM, TreeEvolutionEngine
)


class TestTreeGenomeVM(unittest.TestCase):
    def test_vm_code_add(self):
        tree = TreeGenome(OperationNode('ADD', VariableNode('x'), ConstantNode(3.0)))
        code = tree.to_vm_code()
        
        vm = NeoGlyphVM()
        vm.run(f'PUSH 5\nSTORE a\n{code}')
        result = vm.stack[-1].data[0]
        self.assertAlmostEqual(result, 8.0, places=5)
    
    def test_vm_code_mul(self):
        tree = TreeGenome(OperationNode('MUL', VariableNode('x'), ConstantNode(2.0)))
        code = tree.to_vm_code()
        
        vm = NeoGlyphVM()
        vm.run(f'PUSH 4\nSTORE a\n{code}')
        result = vm.stack[-1].data[0]
        self.assertAlmostEqual(result, 8.0, places=5)
    
    def test_vm_code_2x_plus_1(self):
        x = VariableNode('x')
        tree = TreeGenome(OperationNode('ADD',
            OperationNode('MUL', x, ConstantNode(2.0)),
            ConstantNode(1.0)
        ))
        code = tree.to_vm_code()
        
        for test_x in [-5, -3, -1, 0, 1, 3, 5]:
            vm = NeoGlyphVM()
            vm.run(f'PUSH {test_x}\nSTORE a\n{code}')
            result = vm.stack[-1].data[0]
            expected = 2 * test_x + 1
            self.assertAlmostEqual(result, expected, places=5)
    
    def test_vm_code_sub(self):
        tree = TreeGenome(OperationNode('SUB', VariableNode('x'), ConstantNode(3.0)))
        code = tree.to_vm_code()
        
        vm = NeoGlyphVM()
        vm.run(f'PUSH 10\nSTORE a\n{code}')
        result = vm.stack[-1].data[0]
        self.assertAlmostEqual(result, 7.0, places=5)
    
    def test_vm_code_div(self):
        tree = TreeGenome(OperationNode('DIV', VariableNode('x'), ConstantNode(2.0)))
        code = tree.to_vm_code()
        
        vm = NeoGlyphVM()
        vm.run(f'PUSH 10\nSTORE a\n{code}')
        result = vm.stack[-1].data[0]
        self.assertAlmostEqual(result, 5.0, places=5)
    
    def test_vm_code_sin(self):
        tree = TreeGenome(OperationNode('SIN', VariableNode('x')))
        code = tree.to_vm_code()
        
        vm = NeoGlyphVM()
        vm.run(f'PUSH 0.5\nSTORE a\n{code}')
        result = vm.stack[-1].data[0]
        self.assertAlmostEqual(result, np.sin(0.5), places=5)
    
    def test_vm_code_cos(self):
        tree = TreeGenome(OperationNode('COS', VariableNode('x')))
        code = tree.to_vm_code()
        
        vm = NeoGlyphVM()
        vm.run(f'PUSH 0.5\nSTORE a\n{code}')
        result = vm.stack[-1].data[0]
        self.assertAlmostEqual(result, np.cos(0.5), places=5)
    
    def test_vm_code_exp(self):
        tree = TreeGenome(OperationNode('EXP', VariableNode('x')))
        code = tree.to_vm_code()
        
        vm = NeoGlyphVM()
        vm.run(f'PUSH 1.0\nSTORE a\n{code}')
        result = vm.stack[-1].data[0]
        self.assertAlmostEqual(result, np.exp(1.0), places=5)
    
    def test_vm_code_log(self):
        tree = TreeGenome(OperationNode('LOG', VariableNode('x')))
        code = tree.to_vm_code()
        
        vm = NeoGlyphVM()
        vm.run(f'PUSH 2.0\nSTORE a\n{code}')
        result = vm.stack[-1].data[0]
        self.assertAlmostEqual(result, np.log(2.0), places=5)
    
    def test_vm_code_neg(self):
        tree = TreeGenome(OperationNode('NEG', VariableNode('x')))
        code = tree.to_vm_code()
        
        vm = NeoGlyphVM()
        vm.run(f'PUSH 5.0\nSTORE a\n{code}')
        result = vm.stack[-1].data[0]
        self.assertAlmostEqual(result, -5.0, places=5)


class TestTreeGenomeCrossover(unittest.TestCase):
    def test_crossover_produces_valid_tree(self):
        p1 = TreeGenome(OperationNode('ADD', VariableNode('x'), ConstantNode(1.0)))
        p2 = TreeGenome(OperationNode('MUL', VariableNode('x'), ConstantNode(2.0)))
        child = TreeGenome.crossover(p1, p2)
        self.assertIsNotNone(child.root)
        self.assertIsInstance(child, TreeGenome)
    
    def test_crossover_with_none(self):
        """当一个父本没有操作节点时，直接复制另一个"""
        p1 = TreeGenome(ConstantNode(5.0))
        p2 = TreeGenome(OperationNode('ADD', VariableNode('x'), ConstantNode(1.0)))
        child = TreeGenome.crossover(p1, p2)
        self.assertIsNotNone(child.root)
    
    def test_crossover_preserves_structure(self):
        p1 = TreeGenome(OperationNode('MUL', 
            OperationNode('ADD', VariableNode('x'), ConstantNode(1.0)),
            ConstantNode(3.0)))
        p2 = TreeGenome(OperationNode('SUB', 
            OperationNode('SIN', VariableNode('x')),
            ConstantNode(2.0)))
        child = TreeGenome.crossover(p1, p2)
        self.assertIsNotNone(child.root)
        # 验证子代可以正常评估
        result = child.root.evaluate(2.0)
        self.assertIsInstance(result, (int, float, np.floating))


class TestTreeEvolutionEngine(unittest.TestCase):
    def test_engine_initialization(self):
        engine = TreeEvolutionEngine(pop_size=20, max_depth=2, mutation_rate=0.3)
        engine.initialize_population()
        self.assertEqual(len(engine.population), 20)
    
    def test_engine_evolve_linear(self):
        target_fn = lambda x: x * 2 + 1
        inputs = np.array([-5, -3, -1, 1, 3, 5], dtype=np.float32)
        
        engine = TreeEvolutionEngine(pop_size=30, max_depth=2, mutation_rate=0.3)
        engine.initialize_population()
        best = engine.evolve(inputs, target_fn, generations=60)
        
        self.assertIsNotNone(best)
        self.assertGreater(best.fitness, 0.3,
            f"Should reach fitness > 0.3, got {best.fitness:.4f}")
    
    def test_engine_summary(self):
        target_fn = lambda x: x + 1
        inputs = np.array([-2, 0, 2], dtype=np.float32)
        
        engine = TreeEvolutionEngine(pop_size=10, max_depth=2, mutation_rate=0.3)
        engine.initialize_population()
        engine.evolve(inputs, target_fn, generations=5)
        summary = engine.get_summary()
        
        self.assertIsNotNone(summary)
        self.assertIn('total_generations', summary)
        self.assertIn('final_fitness', summary)
        self.assertEqual(summary['total_generations'], 5)


class TestTreeNodeOperations(unittest.TestCase):
    def test_constant_node(self):
        c = ConstantNode(3.14)
        self.assertAlmostEqual(c.evaluate(0), 3.14, places=5)
        self.assertEqual(c.to_expression(), '3.14')
        self.assertEqual(c.to_vm_code(), 'PUSH 3.14')
    
    def test_variable_node(self):
        v = VariableNode('x')
        self.assertAlmostEqual(v.evaluate(5.0), 5.0, places=5)
        self.assertEqual(v.to_expression(), 'x')
        self.assertEqual(v.to_vm_code(), 'LOAD a')
    
    def test_sin_simplify_zero(self):
        op = OperationNode('SIN', ConstantNode(0.0))
        simplified = op.simplify()
        self.assertAlmostEqual(simplified.evaluate(0), 0.0, places=5)
    
    def test_exp_simplify_zero(self):
        op = OperationNode('EXP', ConstantNode(0.0))
        simplified = op.simplify()
        self.assertAlmostEqual(simplified.evaluate(0), 1.0, places=5)
    
    def test_log_simplify_one(self):
        op = OperationNode('LOG', ConstantNode(1.0))
        simplified = op.simplify()
        self.assertAlmostEqual(simplified.evaluate(0), 0.0, places=5)
    
    def test_neg_simplify_constant(self):
        op = OperationNode('NEG', ConstantNode(5.0))
        simplified = op.simplify()
        self.assertAlmostEqual(simplified.evaluate(0), -5.0, places=5)
    
    def test_like_terms_with_unary(self):
        """SIN(x) + 2*x 不应崩溃，应优雅降级"""
        tree = OperationNode('ADD',
            OperationNode('SIN', VariableNode('x')),
            OperationNode('MUL', VariableNode('x'), ConstantNode(2.0)))
        simplified = tree.simplify()
        self.assertIsNotNone(simplified)


class TestTreeGenomeEvaluateArray(unittest.TestCase):
    def test_perfect_match(self):
        """完美匹配的表达式应该有高 accuracy"""
        tree = TreeGenome(OperationNode('ADD',
            OperationNode('MUL', VariableNode('x'), ConstantNode(2.0)),
            ConstantNode(1.0)))
        X = np.array([1, 2, 3], dtype=np.float64)
        y = 2 * X + 1
        result = tree.evaluate_array(X, y)
        self.assertAlmostEqual(result['accuracy'], 1.0, places=2)
        self.assertLess(result['mse'], 0.001)

    def test_poor_match(self):
        """不匹配的表达式应该有低 accuracy"""
        tree = TreeGenome(ConstantNode(0.0))
        X = np.array([1, 2, 3], dtype=np.float64)
        y = np.array([10, 20, 30], dtype=np.float64)
        result = tree.evaluate_array(X, y)
        self.assertLess(result['accuracy'], 0.3)

    def test_invalid_count(self):
        tree = TreeGenome(OperationNode('ADD', VariableNode('x'), ConstantNode(1.0)))
        X = np.array([1, 2, 3], dtype=np.float64)
        y = np.array([3, 5, 7], dtype=np.float64)
        result = tree.evaluate_array(X, y)
        self.assertEqual(result['valid_count'], 3)


class TestTreeGenomeEvolve(unittest.TestCase):
    def test_can_evolve_linear_function(self):
        target_fn = lambda x: x * 2 + 1
        inputs = np.array([-5, -3, -1, 1, 3, 5], dtype=np.float32)
        
        pop_size = 50
        population = [TreeGenome.create_random(max_depth=2) for _ in range(pop_size)]
        
        best_fitness = 0
        for gen in range(100):
            for genome in population:
                fitness = genome.calculate_fitness(inputs, target_fn)
                if fitness > best_fitness:
                    best_fitness = fitness
            
            population.sort(key=lambda g: g.fitness, reverse=True)
            new_pop = population[:10]
            import random
            for _ in range(pop_size - 10):
                parent = random.choice(population[:25])
                child = parent.copy()
                child.mutate(0.3, fitness=parent.fitness)
                new_pop.append(child)
            population = new_pop
        
        self.assertGreater(best_fitness, 0.5,
            f"Should be able to evolve at least partially, got fitness={best_fitness:.4f}")


class TestEndToEnd(unittest.TestCase):
    def test_evolution_discovers_2x_plus_1_structure(self):
        target_fn = lambda x: x * 2 + 1
        inputs = np.array([-5, -3, -1, 1, 3, 5], dtype=np.float32)
        
        pop_size = 80
        population = [TreeGenome.create_random(max_depth=2) for _ in range(pop_size)]
        
        best_genome = None
        best_fitness = 0
        for gen in range(150):
            for genome in population:
                fitness = genome.calculate_fitness(inputs, target_fn)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_genome = genome.copy()
            
            population.sort(key=lambda g: g.fitness, reverse=True)
            new_pop = population[:16]
            import random
            for _ in range(pop_size - 16):
                parent = random.choice(population[:40])
                child = parent.copy()
                child.mutate(0.25, fitness=parent.fitness)
                new_pop.append(child)
            population = new_pop
        
        simplified = best_genome.simplify()
        expr = simplified.to_expression()
        self.assertGreater(best_fitness, 0.8,
            f"Fitness should be > 0.8, got {best_fitness:.4f}\nBest: {expr}")
        
        error = 0
        for x in inputs:
            pred = best_genome.root.evaluate(x)
            target = target_fn(x)
            error += (pred - target) ** 2
        error /= len(inputs)
        self.assertLess(error, 0.1, f"Error should be small, got {error:.4f}")


if __name__ == '__main__':
    unittest.main()
