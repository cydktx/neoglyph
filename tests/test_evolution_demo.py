import unittest
import json
import os
import sys
import numpy as np

# 添加 examples 目录到 Python 路径
_examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
if _examples_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_examples_dir))


class TestEvolutionDemo(unittest.TestCase):
    def test_test_inputs_outputs(self):
        from evolution_demo import TEST_INPUTS, TARGET_OUTPUTS
        
        self.assertEqual(len(TEST_INPUTS), 11)
        self.assertEqual(len(TARGET_OUTPUTS), 11)
        
        for x, y in zip(TEST_INPUTS, TARGET_OUTPUTS):
            self.assertEqual(y, x * 2 + 1)
    
    def test_evaluate_genome_multipoint(self):
        from evolution_demo import evaluate_genome_multipoint
        from neoglyph.genome import Genome
        
        genome = Genome(length=20)
        
        result = evaluate_genome_multipoint(genome)
        
        self.assertIn('error', result)
        self.assertIn('accuracy', result)
        self.assertIn('valid_count', result)
        self.assertIn('complexity', result)
        self.assertIn('gene_length', result)
    
    def test_calculate_fitness(self):
        from evolution_demo import calculate_fitness
        
        eval_result = {
            'error': 1.0,
            'accuracy': 0.5,
            'valid_count': 11,
            'complexity': 10,
            'gene_length': 30,
            'has_mul': True,
            'has_add': True,
            'has_load': True,
            'push_values': [2.0, 1.0],
            'harmful_instr_count': 0,
            'useful_sequence_count': 2,
            'has_complete_mul_sequence': True,
            'has_complete_add_sequence': True,
            'early_halt_penalty': 0,
            'output_pattern': 'double'
        }
        
        fitness = calculate_fitness(eval_result)
        
        self.assertGreater(fitness, 0)
        self.assertLessEqual(fitness, 3.0)
    
    def test_save_history(self):
        from evolution_demo import save_history
        from neoglyph.genome import Genome
        
        genome = Genome(length=20)
        
        history = []
        for gen in range(1, 6):
            report = {
                'generation': gen,
                'best_fitness': float(gen * 0.1),
                'average_fitness': float(gen * 0.05),
                'improvement': 0,
                'best_genome': genome,
                'complexity': 10,
                'error': float(10 - gen),
                'accuracy': float(gen * 0.05)
            }
            history.append(report)
        
        save_history(history, filename='results/test_history.json')
        
        self.assertTrue(os.path.exists('results/test_history.json'))
        
        with open('results/test_history.json', 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 5)
        for entry in data:
            self.assertIn('generation', entry)
            self.assertIn('best_fitness', entry)
            self.assertIn('average_fitness', entry)
            self.assertIn('script', entry)
        
        os.remove('results/test_history.json')
    
    def test_fitness_improves_over_time(self):
        from evolution_demo import evaluate_genome_multipoint, calculate_fitness, EvolutionEngine, Genome
        
        np.random.seed(42)
        initial_genome = Genome(length=20)
        engine = EvolutionEngine(initial_genome=initial_genome, pop_size=10, mutation_rate=0.3)
        
        fitness_history = []
        for gen in range(1, 11):
            evaluations = []
            for genome in engine.population:
                eval_result = evaluate_genome_multipoint(genome)
                fitness = calculate_fitness(eval_result)
                genome.fitness = fitness
                evaluations.append({'genome': genome, 'fitness': fitness, 'eval_result': eval_result})
            
            evaluations.sort(key=lambda x: x['fitness'], reverse=True)
            best_eval = evaluations[0]
            fitness_history.append(best_eval['fitness'])
            engine.population = engine.generate_next_generation(best_eval, evaluations)
        
        self.assertGreaterEqual(fitness_history[-1], fitness_history[0])
    
    def test_plot_fitness_function_exists(self):
        from evolution_demo import plot_fitness
        
        self.assertIsNotNone(plot_fitness)
    
    def test_elite_selection(self):
        from neoglyph.evolution import EvolutionEngine
        from neoglyph.genome import Genome
        
        np.random.seed(42)
        engine = EvolutionEngine(pop_size=20, mutation_rate=0.1)
        
        evaluations = []
        for i, genome in enumerate(engine.population):
            genome.fitness = float(i * 0.1)
            evaluations.append({
                'genome': genome,
                'fitness': float(i * 0.1)
            })
        
        elite = engine.select_elite(evaluations, elite_ratio=0.1)
        
        self.assertEqual(len(elite), 2)
        self.assertAlmostEqual(elite[0].fitness, 1.9, places=5)
    
    def test_diversity_calculation(self):
        from neoglyph.evolution import EvolutionEngine
        
        engine = EvolutionEngine(pop_size=5, mutation_rate=0.1)
        
        diversity = engine.calculate_diversity(engine.population)
        
        self.assertGreaterEqual(diversity, 0)


if __name__ == '__main__':
    unittest.main()
