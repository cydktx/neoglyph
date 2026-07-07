import unittest
import numpy as np
from neoglyph import TreeGenome, ConstantNode, VariableNode, OperationNode, NeoGlyphVM


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
