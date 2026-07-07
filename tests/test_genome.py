import unittest
import numpy as np
from neoglyph.genome import Genome, GeneticOptimizer


class TestGenome(unittest.TestCase):
    def test_init_random(self):
        g = Genome(length=10)
        self.assertEqual(len(g), 10)
        self.assertIsNone(g.fitness)

    def test_init_with_genes(self):
        genes = [1.0, 2.0, 3.0]
        g = Genome(genes=genes)
        np.testing.assert_array_equal(g.genes, np.array([1., 2., 3.], dtype=np.float32))

    def test_decode_basic(self):
        genes = [1, 2.0, 3.0, 2, 14]
        g = Genome(genes=genes)
        script = g.decode()
        self.assertIn('PUSH', script)
        self.assertIn('ADD', script)
        self.assertIn('HALT', script)

    def test_crossover_single(self):
        g1 = Genome(genes=[1, 2, 3, 4, 5])
        g2 = Genome(genes=[10, 20, 30, 40, 50])
        child = Genome.crossover(g1, g2, method='single')
        self.assertEqual(len(child), 5)
        found_switch = False
        for i in range(1, 5):
            if child.genes[:i].tolist() == g1.genes[:i].tolist() and \
               child.genes[i:].tolist() == g2.genes[i:].tolist():
                found_switch = True
                break
        self.assertTrue(found_switch)

    def test_crossover_uniform(self):
        g1 = Genome(genes=[1, 2, 3])
        g2 = Genome(genes=[10, 20, 30])
        child = Genome.crossover(g1, g2, method='uniform')
        self.assertEqual(len(child), 3)

    def test_mutate(self):
        genes = [1.0, 100.0, 5.0]
        g = Genome(genes=genes.copy())
        g.mutate(mutation_rate=1.0)
        self.assertTrue(np.any(g.genes != np.array(genes, dtype=np.float32)))

    def test_selection_tournament(self):
        g1 = Genome(genes=[1])
        g1.fitness = 0.5
        g2 = Genome(genes=[2])
        g2.fitness = 0.8
        g3 = Genome(genes=[3])
        g3.fitness = 0.3
        selected = Genome.selection([g1, g2, g3], method='tournament')
        self.assertEqual(selected.fitness, 0.8)

    def test_selection_roulette(self):
        g1 = Genome(genes=[1])
        g1.fitness = 0.5
        g2 = Genome(genes=[2])
        g2.fitness = 0.5
        selected = Genome.selection([g1, g2], method='roulette')
        self.assertIn(selected, [g1, g2])

    def test_evaluate(self):
        genes = [1.0, 2.5, 8.0, 100.0, 1.0, 3.5, 8.0, 101.0, 9.0, 100.0, 9.0, 101.0, 2.0, 8.0, 102.0, 14.0]
        g = Genome(genes=genes)

        def target_fn(vm):
            return np.array([6.0])

        fitness = g.evaluate(target_fn)
        self.assertGreater(fitness, 0)

    def test_execute(self):
        genes = [1.0, 2.5, 8.0, 100.0, 1.0, 3.5, 8.0, 101.0, 9.0, 100.0, 9.0, 101.0, 2.0, 8.0, 102.0, 14.0]
        g = Genome(genes=genes)
        vm = g.execute()
        self.assertIn('a', vm.vars)


class TestGeneticOptimizer(unittest.TestCase):
    def test_init(self):
        opt = GeneticOptimizer(pop_size=10, gene_length=5)
        self.assertEqual(len(opt.population), 10)
        self.assertEqual(opt.generation, 0)

    def test_evolve(self):
        opt = GeneticOptimizer(pop_size=5, gene_length=20, mutation_rate=0.1)

        def target_fn(vm):
            return np.array([6.0])

        opt.population[0] = Genome(genes=[1.0, 2.5, 8.0, 100.0, 1.0, 3.5, 8.0, 101.0, 9.0, 100.0, 9.0, 101.0, 2.0, 8.0, 102.0, 14.0])

        best = opt.evolve(target_fn, generations=5)
        self.assertIsInstance(best, Genome)
        self.assertGreater(best.fitness, 0)

    def test_get_best(self):
        opt = GeneticOptimizer(pop_size=5, gene_length=5)
        for i, g in enumerate(opt.population):
            g.fitness = float(i)
        best = opt.get_best()
        self.assertEqual(best.fitness, 4.0)


if __name__ == '__main__':
    unittest.main()
