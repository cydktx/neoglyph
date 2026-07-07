"""
NeoGlyph Advanced Evolution Engine

提升Evolution Core搜索能力：
1. 并行Genome评估
2. Invalid program快速淘汰
3. Curriculum Evolution（分阶段学习）
4. Discovery Score（accuracy + simplicity + generalization）
"""

import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
from .genome import Genome
from .vm import NeoGlyphVM


class DiscoveryScore:
    """Discovery Score计算器
    
    三维度评分：
    - accuracy: 准确度
    - simplicity: 简洁性（更少指令 = 更高分数）
    - generalization: 泛化能力（测试集表现）
    """
    
    def __init__(self, train_inputs, test_inputs, target_fn):
        self.train_inputs = train_inputs
        self.test_inputs = test_inputs
        self.target_fn = target_fn
    
    def calculate(self, genome):
        """计算Discovery Score"""
        # 训练集准确度
        train_accuracy = self._evaluate_accuracy(genome, self.train_inputs)
        
        # 测试集准确度（泛化能力）
        test_accuracy = self._evaluate_accuracy(genome, self.test_inputs)
        
        # 简洁性评分
        simplicity = self._calculate_simplicity(genome)
        
        # 综合Discovery Score
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
        """评估准确度"""
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
        """计算简洁性评分
        
        更少的指令、更短的基因长度 = 更高的简洁性分数
        """
        script = genome.decode()
        lines = [line.strip() for line in script.split('\n') if line.strip()]
        
        # 有效指令数量
        useful_ops = ['LOAD', 'STORE', 'ADD', 'MUL', 'SUB', 'PUSH']
        useful_count = sum(1 for line in lines if any(op in line for op in useful_ops))
        
        # 冗余指令惩罚
        harmful_ops = ['HALT', 'JMP', 'JMP_IF', 'PRINT', 'MATMUL', 'POW', 'NEG']
        harmful_count = sum(1 for line in lines if any(op in line for op in harmful_ops))
        
        # 简洁性计算
        # 最理想情况：3-5条有用指令 = 高分
        ideal_length = 5
        length_penalty = max(0, (useful_count - ideal_length) / 10.0)
        harmful_penalty = harmful_count * 0.2
        
        simplicity = 1.0 - length_penalty - harmful_penalty
        return max(simplicity, 0.0)


class InvalidProgramFilter:
    """Invalid Program快速淘汰过滤器"""
    
    @staticmethod
    def quick_check(genome):
        """快速检查Genome是否无效
        
        返回True表示有效，False表示无效
        """
        # 1. 检查基因长度
        if len(genome.genes) < 3:
            return False
        
        # 2. 检查是否有有效操作
        script = genome.decode()
        lines = [line.strip() for line in script.split('\n') if line.strip()]
        
        useful_ops = ['LOAD', 'ADD', 'MUL', 'SUB']
        has_useful_op = any(any(op in line for op in useful_ops) for line in lines)
        
        if not has_useful_op:
            return False
        
        # 3. 检查是否过早HALT
        if 'HALT' in lines:
            halt_index = lines.index('HALT')
            if halt_index < 2:
                return False
        
        # 4. 快速执行测试
        try:
            vm = genome.execute({'a': 1.0})
            # 检查是否有输出
            has_output = len(vm.vars) > 0 or len(vm.stack) > 0
            return has_output
        except Exception:
            return False
    
    @staticmethod
    def filter_population(population):
        """过滤无效个体，返回有效个体列表"""
        valid_genomes = []
        invalid_count = 0
        
        for genome in population:
            if InvalidProgramFilter.quick_check(genome):
                valid_genomes.append(genome)
            else:
                invalid_count += 1
        
        return valid_genomes, invalid_count


def evaluate_single_genome(args):
    """单个Genome评估函数（用于并行化）"""
    genome, target_fn, inputs = args
    
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
                target = target_fn(float(x))
                error = (actual - target) ** 2
                total_error += error
                valid_count += 1
        except Exception:
            pass
    
    if valid_count == 0:
        return {'genome': genome, 'fitness': 0.0, 'error': float('inf'), 'valid_count': 0}
    
    mse = total_error / valid_count
    accuracy = 1.0 / (1.0 + mse)
    
    # 简洁性评分
    script = genome.decode()
    complexity = len([line for line in script.split('\n') if line.strip()])
    simplicity_penalty = min(complexity / 20.0, 0.3)
    
    fitness = accuracy - simplicity_penalty
    
    return {
        'genome': genome,
        'fitness': max(fitness, 0.001),
        'accuracy': accuracy,
        'error': mse,
        'valid_count': valid_count,
        'complexity': complexity
    }


class ParallelEvaluator:
    """并行Genome评估器"""
    
    def __init__(self, n_workers=None):
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
    
    def evaluate_population(self, population, target_fn, inputs):
        """并行评估整个种群"""
        # 准备参数
        args_list = [(genome, target_fn, inputs) for genome in population]
        
        # 并行评估
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(evaluate_single_genome, args_list))
        
        # 更新Genome fitness
        for result in results:
            result['genome'].fitness = result['fitness']
        
        return results


class CurriculumEvolution:
    """Curriculum Evolution（分阶段学习）
    
    Stage 1: y = x + x (最简单)
    Stage 2: y = 2x + c (系数+常数)
    Stage 3: y = ax + b (一般线性)
    """
    
    STAGES = [
        {
            'name': 'Stage 1: Addition',
            'target_fn': lambda x: x + x,
            'description': 'y = x + x',
            'generations': 100,
            'pop_size': 50,
            'threshold': 0.95  # 达到此fitness后进入下一阶段
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
        """获取当前阶段"""
        return self.STAGES[self.current_stage]
    
    def advance_stage(self, best_fitness):
        """检查是否可以进入下一阶段"""
        current = self.get_current_stage()
        
        if best_fitness >= current['threshold']:
            if self.current_stage < len(self.STAGES) - 1:
                self.current_stage += 1
                print(f"\n🎓 Advancing to {self.STAGES[self.current_stage]['name']}")
                return True
        
        return False
    
    def create_seed_for_stage(self, stage_idx, best_genome_from_previous=None):
        """为当前阶段创建种子基因"""
        stage = self.STAGES[stage_idx]
        
        if best_genome_from_previous is not None and stage_idx > 0:
            # 使用上一阶段的最佳基因作为基础
            genes = best_genome_from_previous.genes.copy()
            
            # 添加一些随机变异
            import random
            for i in range(len(genes)):
                if random.random() < 0.1:
                    genes[i] += random.gauss(0, 0.5)
            
            return Genome(genes=genes)
        
        # 创建新的种子基因
        import random
        genes = []
        
        # 基础操作
        genes.append(9.0)   # LOAD a
        genes.append(100.0) # 变量 a
        
        if stage_idx == 0:
            # Stage 1: x + x
            genes.append(9.0)   # LOAD a
            genes.append(100.0) # 变量 a
            genes.append(2.0)   # ADD
            genes.append(8.0)   # STORE
            genes.append(100.0) # 变量 a
        
        elif stage_idx == 1:
            # Stage 2: 2x + c
            genes.append(9.0)   # LOAD a
            genes.append(100.0) # 变量 a
            genes.append(2.0)   # ADD
            genes.append(1.0)   # PUSH
            genes.append(1.0)   # 常数 1
            genes.append(2.0)   # ADD
            genes.append(8.0)   # STORE
            genes.append(100.0) # 变量 a
        
        elif stage_idx == 2:
            # Stage 3: ax + b
            genes.append(9.0)   # LOAD a
            genes.append(100.0) # 变量 a
            genes.append(3.0)   # MUL
            genes.append(1.0)   # PUSH
            genes.append(1.0)   # 常数
            genes.append(2.0)   # ADD
            genes.append(8.0)   # STORE
            genes.append(100.0) # 变量 a
        
        # 填充剩余基因
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
        """运行当前阶段的进化"""
        stage = self.get_current_stage()
        
        print(f"\n{'='*60}")
        print(f"  {stage['name']}")
        print(f"  Target: {stage['description']}")
        print(f"  Threshold: {stage['threshold']}")
        print(f"{'='*60}")
        
        # 初始化种群
        population = []
        for _ in range(stage['pop_size']):
            genome = Genome(genes=seed_genome.genes.copy())
            population.append(genome)
        
        best_genome = seed_genome
        best_fitness = 0.0
        
        # 进化循环
        for gen in range(1, stage['generations'] + 1):
            # 并行评估
            evaluations = evaluator.evaluate_population(
                population,
                stage['target_fn'],
                self.inputs
            )
            
            # 快速淘汰无效程序
            valid_evaluations = [e for e in evaluations if e['valid_count'] > 0]
            
            if not valid_evaluations:
                continue
            
            # 选择最佳
            valid_evaluations.sort(key=lambda x: x['fitness'], reverse=True)
            best_eval = valid_evaluations[0]
            
            if best_eval['fitness'] > best_fitness:
                best_fitness = best_eval['fitness']
                best_genome = best_eval['genome']
            
            # 进度报告
            if gen % 20 == 0:
                print(f"Gen {gen}: Best={best_fitness:.4f}, Valid={len(valid_evaluations)}")
            
            # 检查是否达标
            if best_fitness >= stage['threshold']:
                print(f"\n✅ Stage completed at Gen {gen}")
                print(f"Best Fitness: {best_fitness:.4f}")
                print(f"Script:\n{best_genome.decode()}")
                break
            
            # 生成下一代（简化版）
            population = self._generate_next_generation(
                population, valid_evaluations, best_genome
            )
        
        return best_genome, best_fitness
    
    def _generate_next_generation(self, population, evaluations, best_genome):
        """生成下一代种群"""
        import random
        
        new_population = []
        
        # 精英保留
        elite_count = int(len(population) * 0.2)
        for e in evaluations[:elite_count]:
            new_population.append(e['genome'])
        
        # 生成新个体
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
        """运行完整的Curriculum Evolution"""
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
            
            # 自动进入下一阶段
            self.current_stage = stage_idx
        
        return best_genome, self.stage_history


class AdvancedEvolutionEngine:
    """高级进化引擎
    
    整合所有改进：
    - 并行评估
    - 快速淘汰
    - Discovery Score
    - Curriculum Evolution
    """
    
    def __init__(self, inputs_train, inputs_test, target_fn):
        self.inputs_train = inputs_train
        self.inputs_test = inputs_test
        self.target_fn = target_fn
        
        self.evaluator = ParallelEvaluator()
        self.discovery_score = DiscoveryScore(inputs_train, inputs_test, target_fn)
        self.filter = InvalidProgramFilter()
    
    def evolve_with_discovery(self, initial_genome, generations=300, pop_size=100):
        """带Discovery Score的进化"""
        print("\n🔬 Advanced Evolution with Discovery Score")
        print(f"Train inputs: {self.inputs_train.tolist()}")
        print(f"Test inputs: {self.inputs_test.tolist()}")
        
        # 初始化种群
        population = []
        for _ in range(pop_size):
            genome = Genome(genes=initial_genome.genes.copy())
            population.append(genome)
        
        best_genome = initial_genome
        best_discovery_score = 0.0
        
        history = []
        
        for gen in range(1, generations + 1):
            # 1. 快速淘汰无效程序
            valid_population, invalid_count = self.filter.filter_population(population)
            
            # 2. 并行评估
            evaluations = self.evaluator.evaluate_population(
                valid_population,
                self.target_fn,
                self.inputs_train
            )
            
            # 3. 计算Discovery Score
            for eval_result in evaluations:
                discovery = self.discovery_score.calculate(eval_result['genome'])
                eval_result['discovery'] = discovery
                eval_result['combined_fitness'] = (
                    eval_result['fitness'] * 0.6 +
                    discovery['discovery_score'] * 0.4
                )
            
            # 4. 选择最佳
            evaluations.sort(key=lambda x: x['combined_fitness'], reverse=True)
            best_eval = evaluations[0]
            
            if best_eval['combined_fitness'] > best_discovery_score:
                best_discovery_score = best_eval['combined_fitness']
                best_genome = best_eval['genome']
            
            # 5. 记录历史
            history.append({
                'generation': gen,
                'best_fitness': best_eval['fitness'],
                'best_discovery_score': best_eval['discovery']['discovery_score'],
                'invalid_count': invalid_count,
                'valid_count': len(valid_population)
            })
            
            # 6. 进度报告
            if gen % 50 == 0:
                print(f"Gen {gen}: Fitness={best_eval['fitness']:.4f}, "
                      f"Discovery={best_eval['discovery']['discovery_score']:.4f}, "
                      f"Valid={len(valid_population)}, Invalid={invalid_count}")
            
            # 7. 生成下一代
            population = self._generate_next_generation(
                population, evaluations, best_genome
            )
        
        # 最终Discovery Score
        final_discovery = self.discovery_score.calculate(best_genome)
        
        print(f"\n{'='*60}")
        print(f"  Final Results")
        print(f"{'='*60}")
        print(f"Discovery Score: {final_discovery['discovery_score']:.4f}")
        print(f"  - Train Accuracy: {final_discovery['train_accuracy']:.4f}")
        print(f"  - Test Accuracy: {final_discovery['test_accuracy']:.4f}")
        print(f"  - Simplicity: {final_discovery['simplicity']:.4f}")
        print(f"  - Generalization: {final_discovery['generalization']:.4f}")
        print(f"\nBest Script:\n{best_genome.decode()}")
        
        return best_genome, final_discovery, history
    
    def _generate_next_generation(self, population, evaluations, best_genome):
        """生成下一代种群"""
        import random
        
        new_population = []
        
        # 精英保留（基于combined_fitness）
        elite_count = int(len(population) * 0.2)
        for e in evaluations[:elite_count]:
            new_population.append(e['genome'])
        
        # 生成新个体
        pool = [e['genome'] for e in evaluations]
        pool_fitness = [(e['genome'], e['combined_fitness']) for e in evaluations]
        
        while len(new_population) < len(population):
            if len(pool) >= 2:
                parent1 = Genome.selection(pool, method='tournament')
                parent2 = Genome.selection(pool, method='tournament')
                
                p1_fitness = next((f for g, f in pool_fitness if g == parent1), 0.0)
                p2_fitness = next((f for g, f in pool_fitness if g == parent2), 0.0)
            else:
                parent1 = best_genome
                parent2 = Genome(genes=best_genome.genes.copy())
                p1_fitness = best_genome.fitness
                p2_fitness = best_genome.fitness
            
            child = Genome.crossover(parent1, parent2)
            child.mutate(0.2, fitness=(p1_fitness + p2_fitness) / 2)
            new_population.append(child)
        
        return new_population[:len(population)]