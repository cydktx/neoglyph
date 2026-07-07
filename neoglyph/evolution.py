import numpy as np
from .genome import Genome
from .vm import NeoGlyphVM


class EvolutionEngine:
    def __init__(self, initial_genome=None, pop_size=5, mutation_rate=0.1):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.generation = 0
        self.history = []
        
        if initial_genome is not None:
            self.population = [initial_genome] + [
                Genome(genes=initial_genome.genes.copy()) 
                for _ in range(pop_size - 1)
            ]
        else:
            self.population = [Genome(length=20) for _ in range(pop_size)]
        
        self.best_genome = None
        self.best_fitness = 0.0

    def evaluate_genome(self, genome, target_fn=None, input_vars=None):
        try:
            vm = genome.execute(input_vars)
            
            report = vm.get_profile_report()
            metrics = vm.get_fitness_metrics()
            
            accuracy = 0.0
            if target_fn is not None:
                result = vm.vars.get('out')
                if result is None and vm.vars:
                    result = list(vm.vars.values())[-1]
                if result is not None:
                    target = target_fn(vm)
                    error = ((result.data - target) ** 2).mean()
                    accuracy = 1.0 / (1.0 + error)
            
            fitness = (
                accuracy * 0.7 +
                metrics['speed'] * 0.1 +
                metrics['memory'] * 0.05 +
                metrics['instruction'] * 0.05 +
                metrics['error'] * 0.1
            )
            
            return {
                'genome': genome,
                'fitness': fitness,
                'accuracy': accuracy,
                'speed': metrics['speed'],
                'memory': metrics['memory'],
                'instruction': metrics['instruction'],
                'error': metrics['error'],
                'report': report
            }
        except Exception:
            return {
                'genome': genome,
                'fitness': 0.0,
                'accuracy': 0.0,
                'speed': 0.0,
                'memory': 0.0,
                'instruction': 0.0,
                'error': 0.0,
                'report': None
            }

    def select(self, evaluations):
        sorted_eval = sorted(evaluations, key=lambda x: x['fitness'], reverse=True)
        return sorted_eval[0]
    
    def calculate_diversity(self, population):
        if len(population) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                genes1 = population[i].genes
                genes2 = population[j].genes
                min_len = min(len(genes1), len(genes2))
                if min_len > 0:
                    diff = np.abs(genes1[:min_len] - genes2[:min_len])
                    total_distance += np.mean(diff)
        
        return total_distance / (len(population) * (len(population) - 1) / 2)

    def select_elite(self, evaluations, elite_ratio=0.1):
        sorted_eval = sorted(evaluations, key=lambda x: x['fitness'], reverse=True)
        elite_count = max(1, int(len(sorted_eval) * elite_ratio))
        return [e['genome'] for e in sorted_eval[:elite_count]]

    def select_diverse(self, evaluations, count=5):
        sorted_eval = sorted(evaluations, key=lambda x: x['fitness'], reverse=True)
        
        selected = [sorted_eval[0]['genome']]
        
        for eval_item in sorted_eval[1:]:
            if len(selected) >= count:
                break
            
            is_diverse = True
            for selected_genome in selected:
                genes1 = eval_item['genome'].genes
                genes2 = selected_genome.genes
                min_len = min(len(genes1), len(genes2))
                if min_len > 0:
                    similarity = np.mean(np.abs(genes1[:min_len] - genes2[:min_len]))
                    if similarity < 0.5:
                        is_diverse = False
                        break
            
            if is_diverse:
                selected.append(eval_item['genome'])
        
        return selected

    def generate_next_generation(self, best_eval, evaluations=None):
        new_population = []
        
        if evaluations:
            elite_genomes = self.select_elite(evaluations, elite_ratio=0.2)
            new_population.extend(elite_genomes)
        elif best_eval:
            elite = best_eval['genome']
            new_population.append(Genome(genes=elite.genes.copy()))
            new_population[0].fitness = elite.fitness
        
        while len(new_population) < self.pop_size:
            pool = []
            pool_with_fitness = []
            if evaluations:
                pool = [e['genome'] for e in evaluations]
                pool_with_fitness = [(e['genome'], e['fitness']) for e in evaluations]
            
            if len(pool) >= 2:
                parent1 = Genome.selection(pool, method='tournament')
                parent2 = Genome.selection(pool, method='tournament')
                
                parent1_fitness = next((f for g, f in pool_with_fitness if g == parent1), 0.0)
                parent2_fitness = next((f for g, f in pool_with_fitness if g == parent2), 0.0)
            elif best_eval:
                parent1 = best_eval['genome']
                parent2 = Genome(genes=parent1.genes.copy())
                parent1_fitness = best_eval['fitness']
                parent2_fitness = best_eval['fitness']
            else:
                parent1 = Genome(length=20)
                parent2 = Genome(length=20)
                parent1_fitness = 0.0
                parent2_fitness = 0.0
            
            child = Genome.crossover(parent1, parent2)
            child_fitness = (parent1_fitness + parent2_fitness) / 2
            child.mutate(self.mutation_rate, fitness=child_fitness)
            new_population.append(child)
        
        return new_population[:self.pop_size]

    def evolve(self, target_fn=None, input_vars=None, generations=10, verbose=False):
        reports = []
        
        for gen in range(generations):
            self.generation = gen + 1
            
            evaluations = []
            for genome in self.population:
                eval_result = self.evaluate_genome(genome, target_fn, input_vars)
                evaluations.append(eval_result)
                genome.fitness = eval_result['fitness']
            
            best_eval = self.select(evaluations)
            
            improvement = 0.0
            if self.best_fitness > 0:
                improvement = ((best_eval['fitness'] - self.best_fitness) / self.best_fitness) * 100
            
            self.best_fitness = best_eval['fitness']
            self.best_genome = best_eval['genome']
            
            report = {
                'generation': self.generation,
                'population': len(self.population),
                'best_fitness': best_eval['fitness'],
                'best_genome': best_eval['genome'],
                'improvement': improvement,
                'evaluations': evaluations
            }
            reports.append(report)
            self.history.append(report)
            
            if verbose:
                self.print_report(report)
            
            self.population = self.generate_next_generation(best_eval)
        
        return reports

    def print_report(self, report):
        improvement_str = f"+{report['improvement']:.1f}%" if report['improvement'] > 0 else \
                          f"{report['improvement']:.1f}%" if report['improvement'] < 0 else "0%"
        
        print(f"Generation {report['generation']}:")
        print(f"  Population: {report['population']}")
        print(f"  Best Fitness: {report['best_fitness']:.4f}")
        print(f"  Improvement: {improvement_str}")
        print(f"  Best Genome: {len(report['best_genome'].genes)} genes")
        print(f"  Decoded Script:")
        print(report['best_genome'].decode())
        print()

    def get_summary(self):
        if not self.history:
            return None
        
        return {
            'total_generations': self.generation,
            'initial_fitness': self.history[0]['best_fitness'],
            'final_fitness': self.best_fitness,
            'best_genome': self.best_genome,
            'average_improvement': sum(r['improvement'] for r in self.history) / len(self.history),
            'history': [
                {'generation': r['generation'], 'fitness': r['best_fitness']}
                for r in self.history
            ]
        }
