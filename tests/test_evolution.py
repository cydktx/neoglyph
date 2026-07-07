import unittest
from neoglyph.evolution import EvolutionEngine
from neoglyph.genome import Genome


class TestEvolutionEngine(unittest.TestCase):
    def test_init(self):
        engine = EvolutionEngine(pop_size=5)
        self.assertEqual(len(engine.population), 5)
        self.assertEqual(engine.generation, 0)
        self.assertIsNone(engine.best_genome)

    def test_init_with_genome(self):
        genome = Genome(genes=[1.0, 2.5, 8.0, 100.0, 14.0])
        engine = EvolutionEngine(initial_genome=genome, pop_size=3)
        self.assertEqual(len(engine.population), 3)
        self.assertEqual(len(engine.population[0].genes), len(genome.genes))

    def test_evaluate_genome(self):
        genome = Genome(genes=[1.0, 2.5, 8.0, 100.0, 1.0, 3.5, 8.0, 101.0, 9.0, 100.0, 9.0, 101.0, 2.0, 8.0, 102.0, 14.0])
        engine = EvolutionEngine()
        
        def target_fn(vm):
            return 6.0
        
        eval_result = engine.evaluate_genome(genome, target_fn)
        self.assertIn('fitness', eval_result)
        self.assertIn('accuracy', eval_result)
        self.assertIn('speed', eval_result)
        self.assertGreater(eval_result['fitness'], 0)

    def test_select(self):
        engine = EvolutionEngine(pop_size=3)
        
        evaluations = [
            {'fitness': 0.5, 'genome': engine.population[0]},
            {'fitness': 0.8, 'genome': engine.population[1]},
            {'fitness': 0.3, 'genome': engine.population[2]}
        ]
        
        best = engine.select(evaluations)
        self.assertEqual(best['fitness'], 0.8)

    def test_generate_next_generation(self):
        genome = Genome(genes=[1.0, 2.5, 8.0, 100.0, 14.0])
        engine = EvolutionEngine(initial_genome=genome, pop_size=5, mutation_rate=0.1)
        
        best_eval = {'genome': genome, 'fitness': 0.5}
        next_gen = engine.generate_next_generation(best_eval)
        
        self.assertEqual(len(next_gen), 5)
        self.assertEqual(next_gen[0].genes.tolist(), genome.genes.tolist())

    def test_evolve_produces_next_generation(self):
        genome = Genome(genes=[1.0, 2.5, 8.0, 100.0, 1.0, 3.5, 8.0, 101.0, 9.0, 100.0, 9.0, 101.0, 2.0, 8.0, 102.0, 14.0])
        engine = EvolutionEngine(initial_genome=genome, pop_size=5, mutation_rate=0.1)
        
        def target_fn(vm):
            return 6.0
        
        reports = engine.evolve(target_fn, generations=3)
        
        self.assertEqual(len(reports), 3)
        self.assertEqual(reports[0]['generation'], 1)
        self.assertEqual(reports[1]['generation'], 2)
        self.assertEqual(reports[2]['generation'], 3)
        
        self.assertIsNotNone(engine.best_genome)
        self.assertGreater(engine.best_fitness, 0)

    def test_report_structure(self):
        genome = Genome(genes=[1.0, 2.5, 8.0, 100.0, 14.0])
        engine = EvolutionEngine(initial_genome=genome, pop_size=3)
        
        def target_fn(vm):
            return 5.0
        
        reports = engine.evolve(target_fn, generations=1)
        
        report = reports[0]
        self.assertIn('generation', report)
        self.assertIn('population', report)
        self.assertIn('best_fitness', report)
        self.assertIn('best_genome', report)
        self.assertIn('improvement', report)
        self.assertIn('evaluations', report)

    def test_improvement_calculation(self):
        genome = Genome(genes=[1.0, 2.5, 8.0, 100.0, 14.0])
        engine = EvolutionEngine(initial_genome=genome, pop_size=3)
        
        def target_fn(vm):
            return 5.0
        
        reports = engine.evolve(target_fn, generations=2)
        
        self.assertIn('improvement', reports[0])
        self.assertIn('improvement', reports[1])

    def test_get_summary(self):
        genome = Genome(genes=[1.0, 2.5, 8.0, 100.0, 14.0])
        engine = EvolutionEngine(initial_genome=genome, pop_size=3)
        
        def target_fn(vm):
            return 5.0
        
        engine.evolve(target_fn, generations=3)
        
        summary = engine.get_summary()
        self.assertIsNotNone(summary)
        self.assertEqual(summary['total_generations'], 3)
        self.assertIn('initial_fitness', summary)
        self.assertIn('final_fitness', summary)
        self.assertIn('best_genome', summary)
        self.assertIn('history', summary)

    def test_empty_evolution(self):
        engine = EvolutionEngine(pop_size=3)
        summary = engine.get_summary()
        self.assertIsNone(summary)

    def test_population_diversity(self):
        genome = Genome(genes=[1.0, 2.5, 8.0, 100.0, 14.0])
        engine = EvolutionEngine(initial_genome=genome, pop_size=5, mutation_rate=0.5)
        
        def target_fn(vm):
            return 5.0
        
        engine.evolve(target_fn, generations=2)
        
        gene_arrays = [g.genes for g in engine.population]
        unique_genomes = len(set(tuple(g) for g in gene_arrays))
        self.assertGreater(unique_genomes, 1)


class TestEvolutionIntegration(unittest.TestCase):
    def test_full_evolution_cycle(self):
        genome = Genome(genes=[1.0, 2.5, 8.0, 100.0, 1.0, 3.5, 8.0, 101.0, 9.0, 100.0, 9.0, 101.0, 2.0, 8.0, 102.0, 14.0])
        engine = EvolutionEngine(initial_genome=genome, pop_size=5, mutation_rate=0.1)
        
        def target_fn(vm):
            return 6.0
        
        reports = engine.evolve(target_fn, generations=5)
        
        for i, report in enumerate(reports):
            self.assertEqual(report['generation'], i + 1)
            self.assertEqual(report['population'], 5)
            self.assertGreaterEqual(report['best_fitness'], 0)

    def test_verbose_output(self):
        genome = Genome(genes=[1.0, 2.5, 8.0, 100.0, 14.0])
        engine = EvolutionEngine(initial_genome=genome, pop_size=3)
        
        def target_fn(vm):
            return 5.0
        
        try:
            reports = engine.evolve(target_fn, generations=1, verbose=True)
            self.assertEqual(len(reports), 1)
        except Exception as e:
            self.fail(f"Verbose evolution raised {e}")


if __name__ == '__main__':
    unittest.main()
