import json
import os
import random
import numpy as np
from neoglyph import EvolutionEngine, Genome


TEST_INPUTS = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=np.float32)
TARGET_OUTPUTS = TEST_INPUTS * 2 + 1


def evaluate_genome_multipoint(genome, target_fn=None):
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
                if target_fn:
                    target = target_fn(float(x))
                else:
                    target = float(x * 2 + 1)
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
    if eval_result['valid_count'] == 0:
        return 0.0
    
    accuracy_score = eval_result['accuracy'] if eval_result['accuracy'] > 0 else 0.0
    
    complexity_penalty = min(eval_result['complexity'] / 15.0, 0.3)
    
    gene_length_penalty = min(eval_result['gene_length'] / 80.0, 0.15)
    
    harmful_penalty = min(eval_result.get('harmful_instr_count', 0) / 4.0, 0.3)
    
    early_halt_penalty = eval_result.get('early_halt_penalty', 0) * 0.5
    
    fitness = accuracy_score - complexity_penalty - gene_length_penalty - harmful_penalty - early_halt_penalty
    
    return max(fitness, 0.001)


def save_history(history, filename='results/evolution_history.json'):
    data = []
    for report in history:
        entry = {
            'generation': int(report['generation']),
            'best_fitness': float(report['best_fitness']),
            'average_fitness': float(report['average_fitness']),
            'improvement': float(report['improvement']),
            'genes': [float(g) for g in report['best_genome'].genes],
            'script': report['best_genome'].decode(),
            'complexity': report['complexity'],
            'error': float(report['error']),
            'accuracy': float(report['accuracy'])
        }
        data.append(entry)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n进化历史已保存到: {filename}")


def plot_fitness(history):
    try:
        import matplotlib.pyplot as plt
        
        generations = [r['generation'] for r in history]
        best_fitness = [r['best_fitness'] for r in history]
        avg_fitness = [r['average_fitness'] for r in history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
        plt.plot(generations, avg_fitness, 'r--', linewidth=1, label='Average Fitness')
        plt.title('Evolution Fitness Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('fitness_curve.png', dpi=150)
        print("Fitness曲线已保存到: fitness_curve.png")
        
        plt.show()
    except ImportError:
        print("\n提示: 安装 matplotlib 以查看图形: pip install matplotlib")


def predict_with_genome(genome, inputs):
    predictions = []
    for x in inputs:
        try:
            vm = genome.execute({'a': float(x)})
            result = vm.vars.get('out')
            if result is None and vm.vars:
                result = list(vm.vars.values())[-1]
            if result is not None:
                predictions.append(float(result.data[0]) if hasattr(result, 'data') else float(result))
            else:
                predictions.append(0.0)
        except Exception:
            predictions.append(0.0)
    return np.array(predictions)


def create_seed_genome():
    genes = []
    
    genes.append(9.0)
    genes.append(100.0)
    genes.append(9.0)
    genes.append(100.0)
    genes.append(2.0)
    genes.append(1.0)
    genes.append(1.0)
    genes.append(2.0)
    genes.append(8.0)
    genes.append(100.0)
    
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


def run_experiment(target_fn, target_name, pop_size=200, generations=500, mutation_rate=0.3):
    print(f"\n{'='*70}")
    print(f"  NeoGlyph Constant Discovery Experiment")
    print(f"  Target: {target_name}")
    print(f"{'='*70}")
    
    target_outputs = np.array([target_fn(float(x)) for x in TEST_INPUTS])
    print(f"测试输入: {TEST_INPUTS.tolist()}")
    print(f"目标输出: {target_outputs.tolist()}")
    print()
    
    np.random.seed(42)
    
    initial_genome = create_seed_genome()
    print("初始基因解码 (包含 LOAD a + LOAD a + ADD 种子):")
    print(initial_genome.decode())
    print()
    
    initial_eval = evaluate_genome_multipoint(initial_genome, target_fn)
    initial_fitness = calculate_fitness(initial_eval)
    print(f"初始评估 - Error: {initial_eval['error']:.4f}, Accuracy: {initial_eval['accuracy']:.4f}")
    print(f"初始Fitness: {initial_fitness:.4f}")
    print()
    
    engine = EvolutionEngine(
        initial_genome=initial_genome,
        pop_size=pop_size,
        mutation_rate=mutation_rate
    )
    
    history = []
    
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
        
        avg_fitness = sum(e['fitness'] for e in evaluations) / len(evaluations)
        
        improvement = 0.0
        if history:
            improvement = ((best_eval['fitness'] - history[-1]['best_fitness']) / 
                           max(history[-1]['best_fitness'], 0.0001)) * 100
        
        report = {
            'generation': gen,
            'best_fitness': best_eval['fitness'],
            'average_fitness': avg_fitness,
            'improvement': improvement,
            'best_genome': best_eval['genome'],
            'complexity': best_eval['eval_result']['complexity'],
            'error': best_eval['eval_result']['error'],
            'accuracy': best_eval['eval_result']['accuracy']
        }
        history.append(report)
        engine.best_genome = best_eval['genome']
        engine.best_fitness = best_eval['fitness']
        engine.generation = gen
        
        if gen in [1, 100, 500]:
            improvement_str = f"+{improvement:.1f}%" if improvement > 0 else \
                              f"{improvement:.1f}%" if improvement < 0 else "0%"
            print(f"\n{'='*50}")
            print(f"Generation {gen}:")
            print(f"{'='*50}")
            print(f"  Best Fitness: {best_eval['fitness']:.6f}")
            print(f"  Average Fitness: {avg_fitness:.6f}")
            print(f"  Improvement: {improvement_str}")
            print(f"  Error: {best_eval['eval_result']['error']:.6f}")
            print(f"  Accuracy: {best_eval['eval_result']['accuracy']:.6f}")
            print(f"  Complexity: {best_eval['eval_result']['complexity']} lines")
            print(f"\n  Decoded Script:")
            print(best_eval['genome'].decode())
            print()
        
        if gen % 100 == 0:
            print(f"Generation {gen}: Best Fitness = {best_eval['fitness']:.4f}, Error = {best_eval['eval_result']['error']:.4f}")
        
        engine.population = engine.generate_next_generation(best_eval, evaluations)
    
    final_genome = engine.best_genome
    best_history_genome = None
    best_history_fitness = 0.0
    for report in history:
        if report['best_fitness'] > best_history_fitness:
            best_history_fitness = report['best_fitness']
            best_history_genome = report['best_genome']
    
    if best_history_genome and best_history_fitness > final_genome.fitness:
        print(f"\n⚠️  发现历史最佳个体 (Fitness: {best_history_fitness:.4f}) 优于最终个体，使用历史最佳")
        final_genome = best_history_genome
    
    predictions = predict_with_genome(final_genome, TEST_INPUTS)
    
    print("\n" + "="*60)
    print("实验结束 - 最终结果")
    print("="*60)
    print("\nBest Genome:")
    print(f"  基因长度: {len(final_genome.genes)}")
    print(f"  Fitness: {engine.best_fitness:.6f}")
    print("\nDecoded NeoGlyph Script:")
    print(final_genome.decode())
    print("\nPrediction vs Target:")
    print(f"{'Input':>6} {'Prediction':>12} {'Target':>10} {'Error':>10}")
    print("-" * 40)
    
    total_error = 0.0
    for x, pred, target in zip(TEST_INPUTS, predictions, target_outputs):
        error = abs(pred - target)
        total_error += error
        print(f"{x:>6.0f} {pred:>12.6f} {target:>10.0f} {error:>10.6f}")
    
    avg_error = total_error / len(TEST_INPUTS)
    print("-" * 40)
    print(f"{'Avg':>6} {'':>12} {'':>10} {avg_error:>10.6f}")
    print(f"\n✅ 平均误差: {avg_error:.6f}")
    
    return history, final_genome, avg_error


def main():
    experiments = [
        {
            'name': 'y = x + 3',
            'fn': lambda x: x + 3
        },
        {
            'name': 'y = 2x + 1',
            'fn': lambda x: x * 2 + 1
        }
    ]
    
    all_results = []
    
    for exp in experiments:
        history, genome, avg_error = run_experiment(
            target_fn=exp['fn'],
            target_name=exp['name'],
            pop_size=300,
            generations=1000,
            mutation_rate=0.2
        )
        
        all_results.append({
            'experiment': exp['name'],
            'history': history,
            'genome': genome,
            'avg_error': avg_error
        })
        
        save_history(history, filename=f'results/evolution_{exp["name"].replace(" ", "_")}.json')
        plot_fitness(history)
    
    print("\n" + "="*70)
    print("  实验总结")
    print("="*70)
    for result in all_results:
        print(f"\n实验: {result['experiment']}")
        print(f"  最终平均误差: {result['avg_error']:.6f}")
        print(f"  最佳脚本:")
        print(result['genome'].decode())
    
    return all_results


if __name__ == '__main__':
    main()
