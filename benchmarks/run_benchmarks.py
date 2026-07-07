#!/usr/bin/env python3
"""
NeoGlyph Benchmark System

测试NeoGlyph在不同问题类型上的进化能力：
- Linear: 线性函数
- Polynomial: 多项式函数
- Control: 控制流（条件判断）
"""

import json
import os
import sys
import random
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neoglyph import Genome


TEST_INPUTS = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=np.float32)

# Train/Test分离
TRAIN_INPUTS = np.array([-5, -3, -1, 1, 3, 5], dtype=np.float32)  # 6个训练点
TEST_INPUTS_FULL = np.array([-4, -2, 0, 2, 4], dtype=np.float32)   # 5个测试点


def create_seed_genome_for_type(problem_type='linear'):
    """根据问题类型创建种子基因"""
    genes = []
    
    if problem_type == 'linear':
        # 线性函数种子：LOAD + ADD + PUSH
        genes.append(9.0)   # LOAD a
        genes.append(100.0) # 变量 a
        genes.append(9.0)   # LOAD a
        genes.append(100.0) # 变量 a
        genes.append(2.0)   # ADD
        genes.append(1.0)   # PUSH
        genes.append(1.0)   # 常数 1
        genes.append(2.0)   # ADD
        genes.append(8.0)   # STORE a
        genes.append(100.0) # 变量 a
    
    elif problem_type == 'polynomial':
        # 多项式种子：LOAD + LOAD + MUL + ADD
        genes.append(9.0)   # LOAD a
        genes.append(100.0) # 变量 a
        genes.append(9.0)   # LOAD a
        genes.append(100.0) # 变量 a
        genes.append(3.0)   # MUL
        genes.append(9.0)   # LOAD a
        genes.append(100.0) # 变量 a
        genes.append(2.0)   # ADD
        genes.append(8.0)   # STORE a
        genes.append(100.0) # 变量 a
    
    elif problem_type == 'control':
        # 控制流种子：LOAD + JMP_IF
        genes.append(9.0)   # LOAD a
        genes.append(100.0) # 变量 a
        genes.append(1.0)   # PUSH
        genes.append(0.0)   # 常数 0
        genes.append(14.0)  # JMP_IF
        genes.append(100.0) # 变量 a
    
    # 填充剩余基因
    remaining = 50 - len(genes)
    for _ in range(remaining):
        rand = random.random()
        if rand < 0.20:
            ops = [1, 2, 3, 9, 8]
            genes.append(float(random.choice(ops)))
        elif rand < 0.30:
            genes.append(float(random.randint(100, 110)))
        elif rand < 0.50:
            genes.append(random.uniform(0.5, 2.5))
        else:
            val = random.uniform(-3, 3)
            while val.is_integer():
                val = random.uniform(-3, 3)
            genes.append(val)
    
    return Genome(genes=genes)


def evaluate_genome_multipoint(genome, target_fn):
    """多点评估Genome"""
    total_error = 0.0
    valid_count = 0
    
    for x in TEST_INPUTS:
        try:
            vm = genome.execute({'a': float(x)})
            result = vm.vars.get('out')
            if result is None and vm.vars:
                result = list(vm.vars.values())[-1]
            if result is not None:
                actual = float(result.data[0]) if hasattr(result, 'data') else float(result)
                target = target_fn(float(x))
                total_error += (actual - target) ** 2
                valid_count += 1
        except Exception:
            pass
    
    script = genome.decode()
    lines = script.split('\n')
    complexity = len(lines)
    
    harmful_instr_count = sum(1 for line in lines if line in ['HALT', 'JMP a', 'JMP_IF a'])
    
    halt_count = sum(1 for line in lines if line == 'HALT')
    early_halt_penalty = 0
    if halt_count > 0:
        first_halt_idx = lines.index('HALT') if 'HALT' in lines else len(lines)
        useful_before_halt = sum(1 for line in lines[:first_halt_idx] if line in ['LOAD a', 'MUL', 'ADD', 'STORE a'])
        if useful_before_halt < 2:
            early_halt_penalty = 1
    
    return {
        'error': float('inf') if valid_count == 0 else total_error / valid_count,
        'accuracy': 0.0 if valid_count == 0 else 1.0 / (1.0 + total_error / valid_count),
        'valid_count': valid_count,
        'complexity': complexity,
        'gene_length': len(genome.genes),
        'harmful_instr_count': harmful_instr_count,
        'early_halt_penalty': early_halt_penalty
    }


def calculate_fitness(eval_result):
    """计算Fitness"""
    if eval_result['valid_count'] == 0:
        return 0.0
    
    accuracy_score = eval_result['accuracy'] if eval_result['accuracy'] > 0 else 0.0
    
    complexity_penalty = min(eval_result['complexity'] / 15.0, 0.3)
    
    gene_length_penalty = min(eval_result['gene_length'] / 80.0, 0.15)
    
    harmful_penalty = min(eval_result.get('harmful_instr_count', 0) / 4.0, 0.3)
    
    early_halt_penalty = eval_result.get('early_halt_penalty', 0) * 0.5
    
    fitness = accuracy_score - complexity_penalty - gene_length_penalty - harmful_penalty - early_halt_penalty
    
    return max(fitness, 0.001)


def predict_with_genome(genome, inputs):
    """使用Genome预测"""
    predictions = []
    for x in inputs:
        try:
            vm = genome.execute({'a': float(x)})
            result = vm.vars.get('out')
            if result is None and vm.vars:
                result = list(vm.vars.values())[-1]
            if result is not None:
                actual = float(result.data[0]) if hasattr(result, 'data') else float(result)
                predictions.append(actual)
            else:
                predictions.append(0.0)
        except Exception:
            predictions.append(0.0)
    return np.array(predictions)


class SimpleEvolutionEngine:
    """简化版进化引擎，用于Benchmark"""
    
    def __init__(self, initial_genome, pop_size=100, mutation_rate=0.2):
        self.initial_genome = initial_genome
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.best_genome = None
        self.best_fitness = 0.0
        self.generation = 0
        
        # 初始化种群
        for _ in range(pop_size):
            genome = Genome(genes=initial_genome.genes.copy())
            self.population.append(genome)
    
    def generate_next_generation(self, best_eval, evaluations):
        """生成下一代"""
        new_population = []
        
        # 精英保留 (20%)
        sorted_eval = sorted(evaluations, key=lambda x: x['fitness'], reverse=True)
        elite_count = max(1, int(len(sorted_eval) * 0.2))
        for e in sorted_eval[:elite_count]:
            new_population.append(e['genome'])
        
        # 生成新个体
        pool = [e['genome'] for e in evaluations]
        pool_with_fitness = [(e['genome'], e['fitness']) for e in evaluations]
        
        while len(new_population) < self.pop_size:
            # 锦标赛选择
            parent1 = Genome.selection(pool, method='tournament')
            parent2 = Genome.selection(pool, method='tournament')
            
            parent1_fitness = next((f for g, f in pool_with_fitness if g == parent1), 0.0)
            parent2_fitness = next((f for g, f in pool_with_fitness if g == parent2), 0.0)
            
            # 交叉
            child = Genome.crossover(parent1, parent2)
            
            # 变异（带fitness保护）
            child_fitness = (parent1_fitness + parent2_fitness) / 2
            child.mutate(self.mutation_rate, fitness=child_fitness)
            
            new_population.append(child)
        
        return new_population[:self.pop_size]


def run_single_benchmark(name, target_fn, problem_type, generations=300, pop_size=100):
    """运行单个benchmark"""
    print(f"\n{'='*60}")
    print(f"  Benchmark: {name}")
    print(f"  Type: {problem_type}")
    print(f"{'='*60}")
    
    target_outputs = np.array([target_fn(float(x)) for x in TEST_INPUTS])
    print(f"Target outputs: {target_outputs.tolist()}")
    
    np.random.seed(42)
    random.seed(42)
    
    # 创建种子基因
    initial_genome = create_seed_genome_for_type(problem_type)
    print(f"\nInitial script:\n{initial_genome.decode()}")
    
    # 初始评估
    initial_eval = evaluate_genome_multipoint(initial_genome, target_fn)
    initial_fitness = calculate_fitness(initial_eval)
    print(f"\nInitial fitness: {initial_fitness:.4f}")
    print(f"Initial error: {initial_eval['error']:.4f}")
    
    # 进化
    engine = SimpleEvolutionEngine(initial_genome, pop_size=pop_size, mutation_rate=0.2)
    
    history = []
    best_genome_overall = initial_genome
    best_fitness_overall = initial_fitness
    
    for gen in range(1, generations + 1):
        evaluations = []
        for genome in engine.population:
            eval_result = evaluate_genome_multipoint(genome, target_fn)
            fitness = calculate_fitness(eval_result)
            genome.fitness = fitness
            evaluations.append({
                'genome': genome,
                'fitness': fitness,
                'eval_result': eval_result
            })
        
        evaluations.sort(key=lambda x: x['fitness'], reverse=True)
        best_eval = evaluations[0]
        
        # 更新全局最佳
        if best_eval['fitness'] > best_fitness_overall:
            best_fitness_overall = best_eval['fitness']
            best_genome_overall = best_eval['genome']
        
        avg_fitness = sum(e['fitness'] for e in evaluations) / len(evaluations)
        
        if gen % 50 == 0:
            print(f"Gen {gen}: Best={best_eval['fitness']:.4f}, Avg={avg_fitness:.4f}, Error={best_eval['eval_result']['error']:.4f}")
        
        history.append({
            'generation': gen,
            'best_fitness': best_eval['fitness'],
            'average_fitness': avg_fitness,
            'error': best_eval['eval_result']['error']
        })
        
        engine.population = engine.generate_next_generation(best_eval, evaluations)
    
    # 最终评估
    final_eval = evaluate_genome_multipoint(best_genome_overall, target_fn)
    predictions = predict_with_genome(best_genome_overall, TEST_INPUTS)
    
    total_error = 0.0
    for pred, target in zip(predictions, target_outputs):
        total_error += abs(pred - target)
    avg_error = total_error / len(TEST_INPUTS)
    
    print(f"\n{'='*60}")
    print(f"  Final Results")
    print(f"{'='*60}")
    print(f"Best fitness: {best_fitness_overall:.4f}")
    print(f"Average error: {avg_error:.6f}")
    print(f"\nBest script:\n{best_genome_overall.decode()}")
    
    print(f"\nPredictions vs Targets:")
    print(f"{'Input':>6} {'Pred':>10} {'Target':>10} {'Error':>8}")
    print("-" * 40)
    for x, pred, target in zip(TEST_INPUTS, predictions, target_outputs):
        error = abs(pred - target)
        print(f"{x:>6.0f} {pred:>10.4f} {target:>10.0f} {error:>8.4f}")
    
    # 返回benchmark结果
    return {
        'name': name,
        'type': problem_type,
        'target_outputs': target_outputs.tolist(),
        'generations': generations,
        'population_size': pop_size,
        'initial': {
            'fitness': initial_fitness,
            'error': initial_eval['error'],
            'script': initial_genome.decode(),
            'genes': initial_genome.genes.tolist()
        },
        'final': {
            'fitness': best_fitness_overall,
            'error': avg_error,
            'script': best_genome_overall.decode(),
            'genes': best_genome_overall.genes.tolist(),
            'predictions': predictions.tolist()
        },
        'history': history
    }


def main():
    """运行所有benchmark"""
    print("\n" + "="*70)
    print("  NeoGlyph Benchmark System")
    print("  Testing Evolution Capabilities")
    print("="*70)
    
    benchmarks = []
    
    # Linear benchmarks
    linear_tests = [
        ('y = x + 3', lambda x: x + 3, 'linear'),
        ('y = 2x + 1', lambda x: x * 2 + 1, 'linear'),
        ('y = 5x - 7', lambda x: x * 5 - 7, 'linear')
    ]
    
    # Polynomial benchmarks
    polynomial_tests = [
        ('y = x*x', lambda x: x * x, 'polynomial'),
        ('y = x*x + 2x + 1', lambda x: x * x + 2 * x + 1, 'polynomial')
    ]
    
    # Control benchmarks
    control_tests = [
        ('if x>0: y=1 else y=0', lambda x: 1.0 if x > 0 else 0.0, 'control')
    ]
    
    # Run linear benchmarks
    print("\n" + "="*70)
    print("  LINEAR BENCHMARKS")
    print("="*70)
    for name, fn, type_ in linear_tests:
        result = run_single_benchmark(name, fn, type_, generations=300, pop_size=100)
        benchmarks.append(result)
    
    # Run polynomial benchmarks
    print("\n" + "="*70)
    print("  POLYNOMIAL BENCHMARKS")
    print("="*70)
    for name, fn, type_ in polynomial_tests:
        result = run_single_benchmark(name, fn, type_, generations=500, pop_size=150)
        benchmarks.append(result)
    
    # Run control benchmarks
    print("\n" + "="*70)
    print("  CONTROL FLOW BENCHMARKS")
    print("="*70)
    for name, fn, type_ in control_tests:
        result = run_single_benchmark(name, fn, type_, generations=500, pop_size=150)
        benchmarks.append(result)
    
    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'version': 'NeoGlyph v3.2',
        'test_inputs': TEST_INPUTS.tolist(),
        'total_benchmarks': len(benchmarks),
        'benchmarks': benchmarks,
        'summary': {}
    }
    
    # Calculate summary
    for type_name in ['linear', 'polynomial', 'control']:
        type_results = [b for b in benchmarks if b['type'] == type_name]
        if type_results:
            avg_initial_fitness = np.mean([b['initial']['fitness'] for b in type_results])
            avg_final_fitness = np.mean([b['final']['fitness'] for b in type_results])
            avg_final_error = np.mean([b['final']['error'] for b in type_results])
            
            report['summary'][type_name] = {
                'count': len(type_results),
                'avg_initial_fitness': avg_initial_fitness,
                'avg_final_fitness': avg_final_fitness,
                'avg_final_error': avg_final_error,
                'improvement': avg_final_fitness - avg_initial_fitness
            }
    
    # Save report
    report_path = 'benchmarks/benchmark_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"\nReport saved to: {report_path}")
    
    # Print summary
    print(f"\nSummary:")
    for type_name, stats in report['summary'].items():
        print(f"\n  {type_name.upper()}:")
        print(f"    Benchmarks: {stats['count']}")
        print(f"    Avg Initial Fitness: {stats['avg_initial_fitness']:.4f}")
        print(f"    Avg Final Fitness: {stats['avg_final_fitness']:.4f}")
        print(f"    Avg Final Error: {stats['avg_final_error']:.6f}")
        print(f"    Improvement: {stats['improvement']:.4f}")
    
    return report


if __name__ == '__main__':
    main()