"""
NeoGlyph v4 Evolution Advanced - 可配置策略

所有高级功能已改造为可配置策略，不再创建新的 Engine 类。
主 Engine 统一使用: neoglyph.evolution.EvolutionEngine

保留的独立策略类（向后兼容）：
- DiscoveryScore: 发现评分（三维度：accuracy + simplicity + generalization）
- InvalidProgramFilter: 无效程序快速淘汰
- CurriculumEvolution: 分阶段课程学习
"""

import numpy as np
from .genome import Genome
from .evolution import EvolutionEngine, ParallelEvaluator


class DiscoveryScore:
    """Discovery Score 计算器（向后兼容）

    三维度评分：
    - accuracy: 准确度
    - simplicity: 简洁性
    - generalization: 泛化能力
    """

    def __init__(self, train_inputs, test_inputs, target_fn):
        self.train_inputs = train_inputs
        self.test_inputs = test_inputs
        self.target_fn = target_fn

    def calculate(self, genome):
        train_accuracy = self._evaluate_accuracy(genome, self.train_inputs)
        test_accuracy = self._evaluate_accuracy(genome, self.test_inputs)
        simplicity = self._calculate_simplicity(genome)

        discovery_score = (
            train_accuracy * 0.4 +
            test_accuracy * 0.3 +
            simplicity * 0.3
        )

        return {
            'discovery_score': discovery_score,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'simplicity': simplicity,
            'generalization': test_accuracy / max(train_accuracy, 0.001)
        }

    def _evaluate_accuracy(self, genome, inputs):
        total_error = 0.0
        valid_count = 0

        for x in inputs:
            try:
                vm = genome.execute({'a': float(x)})
                result = vm.vars.get('out')
                if result is None and vm.vars:
                    result = list(vm.vars.values())[-1]

                if result is not None:
                    actual = float(result.data[0]) if hasattr(result, 'data') else float(result)
                    target = self.target_fn(float(x))
                    error = (actual - target) ** 2
                    total_error += error
                    valid_count += 1
            except Exception:
                pass

        if valid_count == 0:
            return 0.0
        mse = total_error / valid_count
        accuracy = 1.0 / (1.0 + mse)
        return max(accuracy, 0.0)

    def _calculate_simplicity(self, genome):
        script = genome.decode()
        lines = [line.strip() for line in script.split('\n') if line.strip()]
        useful_ops = ['LOAD', 'STORE', 'ADD', 'MUL', 'SUB', 'PUSH']
        useful_count = sum(1 for line in lines if any(op in line for op in useful_ops))
        harmful_ops = ['HALT', 'JMP', 'JMP_IF', 'PRINT', 'MATMUL', 'POW', 'NEG']
        harmful_count = sum(1 for line in lines if any(op in line for op in harmful_ops))
        ideal_length = 5
        length_penalty = max(0, (useful_count - ideal_length) / 10.0)
        harmful_penalty = harmful_count * 0.2
        simplicity = 1.0 - length_penalty - harmful_penalty
        return max(simplicity, 0.0)


class InvalidProgramFilter:
    """Invalid Program 快速淘汰过滤器（向后兼容）"""

    @staticmethod
    def quick_check(genome):
        if len(genome.genes) < 3:
            return False
        script = genome.decode()
        lines = [line.strip() for line in script.split('\n') if line.strip()]
        useful_ops = ['LOAD', 'ADD', 'MUL', 'SUB']
        has_useful_op = any(any(op in line for op in useful_ops) for line in lines)
        if not has_useful_op:
            return False
        if 'HALT' in lines:
            halt_index = lines.index('HALT')
            if halt_index < 2:
                return False
        try:
            vm = genome.execute({'a': 1.0})
            has_output = len(vm.vars) > 0 or len(vm.stack) > 0
            return has_output
        except Exception:
            return False

    @staticmethod
    def filter_population(population):
        valid_genomes = []
        invalid_count = 0
        for genome in population:
            if InvalidProgramFilter.quick_check(genome):
                valid_genomes.append(genome)
            else:
                invalid_count += 1
        return valid_genomes, invalid_count


class CurriculumEvolution:
    """Curriculum Evolution 分阶段学习（向后兼容）"""

    STAGES = [
        {
            'name': 'Stage 1: Addition',
            'target_fn': lambda x: x + x,
            'description': 'y = x + x',
            'generations': 100,
            'pop_size': 50,
            'threshold': 0.95
        },
        {
            'name': 'Stage 2: Coefficient + Constant',
            'target_fn': lambda x: x * 2 + 1,
            'description': 'y = 2x + 1',
            'generations': 200,
            'pop_size': 100,
            'threshold': 0.90
        },
        {
            'name': 'Stage 3: General Linear',
            'target_fn': lambda x: x * 5 - 7,
            'description': 'y = 5x - 7',
            'generations': 300,
            'pop_size': 150,
            'threshold': 0.80
        }
    ]

    def __init__(self, inputs):
        self.inputs = inputs
        self.current_stage = 0
        self.stage_history = []

    def get_current_stage(self):
        return self.STAGES[self.current_stage]

    def advance_stage(self, best_fitness):
        current = self.get_current_stage()
        if best_fitness >= current['threshold']:
            if self.current_stage < len(self.STAGES) - 1:
                self.current_stage += 1
                print(f"\nAdvancing to {self.STAGES[self.current_stage]['name']}")
                return True
        return False

    def create_seed_for_stage(self, stage_idx, best_genome_from_previous=None):
        import random
        stage = self.STAGES[stage_idx]
        if best_genome_from_previous is not None and stage_idx > 0:
            genes = best_genome_from_previous.genes.copy()
            for i in range(len(genes)):
                if random.random() < 0.1:
                    genes[i] += random.gauss(0, 0.5)
            return Genome(genes=genes)
        genes = [9.0, 100.0]
        if stage_idx == 0:
            genes.extend([9.0, 100.0, 2.0, 8.0, 100.0])
        elif stage_idx == 1:
            genes.extend([9.0, 100.0, 2.0, 1.0, 1.0, 2.0, 8.0, 100.0])
        elif stage_idx == 2:
            genes.extend([9.0, 100.0, 3.0, 1.0, 1.0, 2.0, 8.0, 100.0])
        remaining = 50 - len(genes)
        for _ in range(remaining):
            rand = random.random()
            if rand < 0.2:
                genes.append(float(random.randint(1, 20)))
            elif rand < 0.3:
                genes.append(float(random.randint(100, 110)))
            else:
                genes.append(random.uniform(-3, 3))
        return Genome(genes=genes)

    def run_stage(self, evaluator, seed_genome):
        import random
        stage = self.get_current_stage()
        print(f"\n{'='*60}")
        print(f"  {stage['name']}")
        print(f"  Target: {stage['description']}")
        print(f"  Threshold: {stage['threshold']}")
        print(f"{'='*60}")

        population = []
        for _ in range(stage['pop_size']):
            genome = Genome(genes=seed_genome.genes.copy())
            population.append(genome)

        best_genome = seed_genome
        best_fitness = 0.0

        for gen in range(1, stage['generations'] + 1):
            evaluations = evaluator.evaluate_population(
                population, stage['target_fn'], self.inputs)
            valid_evaluations = [e for e in evaluations if e['valid_count'] > 0]
            if not valid_evaluations:
                continue
            valid_evaluations.sort(key=lambda x: x['fitness'], reverse=True)
            best_eval = valid_evaluations[0]
            if best_eval['fitness'] > best_fitness:
                best_fitness = best_eval['fitness']
                best_genome = best_eval['genome']
            if gen % 20 == 0:
                print(f"Gen {gen}: Best={best_fitness:.4f}, Valid={len(valid_evaluations)}")
            if best_fitness >= stage['threshold']:
                print(f"\nStage completed at Gen {gen}")
                print(f"Best Fitness: {best_fitness:.4f}")
                break
            population = self._generate_next_generation(
                population, valid_evaluations, best_genome)
        return best_genome, best_fitness

    def _generate_next_generation(self, population, evaluations, best_genome):
        import random
        new_population = []
        elite_count = int(len(population) * 0.2)
        for e in evaluations[:elite_count]:
            new_population.append(e['genome'])
        pool = [e['genome'] for e in evaluations]
        while len(new_population) < len(population):
            if len(pool) >= 2:
                parent1 = Genome.selection(pool, method='tournament')
                parent2 = Genome.selection(pool, method='tournament')
            else:
                parent1 = best_genome
                parent2 = Genome(genes=best_genome.genes.copy())
            child = Genome.crossover(parent1, parent2)
            child.mutate(0.2, fitness=(parent1.fitness + parent2.fitness) / 2)
            new_population.append(child)
        return new_population[:len(population)]

    def run_full_curriculum(self):
        evaluator = ParallelEvaluator()
        best_genome = None
        for stage_idx in range(len(self.STAGES)):
            seed_genome = self.create_seed_for_stage(stage_idx, best_genome)
            best_genome, best_fitness = self.run_stage(evaluator, seed_genome)
            self.stage_history.append({
                'stage': self.STAGES[stage_idx]['name'],
                'best_fitness': best_fitness,
                'best_genome': best_genome
            })
            self.current_stage = stage_idx
        return best_genome, self.stage_history


# 向后兼容别名：AdvancedEvolutionEngine 已废弃，保留为工厂函数
def AdvancedEvolutionEngine(inputs_train, inputs_test, target_fn):
    """向后兼容：返回 EvolutionEngine(genome_type="linear")"""
    return EvolutionEngine(genome_type="linear")