#!/usr/bin/env python3
"""
NeoGlyph Advanced Evolution Demo

演示高级进化特性：
1. 并行Genome评估
2. Invalid program快速淘汰
3. Curriculum Evolution
4. Discovery Score
5. Train/Test分离
"""

import numpy as np
from neoglyph import (
    Genome,
    CurriculumEvolution,
    AdvancedEvolutionEngine,
    DiscoveryScore,
    InvalidProgramFilter,
    ParallelEvaluator
)


# 定义测试数据
TRAIN_INPUTS = np.array([-5, -3, -1, 1, 3, 5], dtype=np.float32)
TEST_INPUTS = np.array([-4, -2, 0, 2, 4], dtype=np.float32)


def demo_invalid_program_filter():
    """演示Invalid Program快速淘汰"""
    print("\n" + "="*60)
    print("  Invalid Program Filter Demo")
    print("="*60)
    
    # 创建一些基因组
    genomes = []
    
    # 有效Genome
    valid_genes = [9.0, 100.0, 9.0, 100.0, 2.0, 8.0, 100.0]  # LOAD a + LOAD a + ADD + STORE a
    genomes.append(Genome(genes=valid_genes))
    
    # 无效Genome - 过短
    genomes.append(Genome(genes=[1.0, 2.0]))
    
    # 无效Genome - 过早HALT
    invalid_genes = [5.0, 9.0, 100.0]  # HALT + LOAD a
    genomes.append(Genome(genes=invalid_genes))
    
    # 添加更多随机Genome
    for _ in range(5):
        genomes.append(Genome(length=30))
    
    print(f"Created {len(genomes)} genomes")
    
    # 过滤无效程序
    valid_genomes, invalid_count = InvalidProgramFilter.filter_population(genomes)
    
    print(f"Valid genomes: {len(valid_genomes)}")
    print(f"Invalid genomes: {invalid_count}")
    
    for i, genome in enumerate(valid_genomes):
        print(f"\nValid Genome {i+1}:")
        print(genome.decode())


def demo_discovery_score():
    """演示Discovery Score计算"""
    print("\n" + "="*60)
    print("  Discovery Score Demo")
    print("="*60)
    
    target_fn = lambda x: x * 2 + 1
    
    discovery_score = DiscoveryScore(TRAIN_INPUTS, TEST_INPUTS, target_fn)
    
    # 创建一个简洁的Genome
    genes = [
        9.0, 100.0,  # LOAD a
        9.0, 100.0,  # LOAD a
        2.0,         # ADD
        1.0, 1.0,    # PUSH 1
        2.0,         # ADD
        8.0, 100.0   # STORE a
    ]
    
    # 填充剩余基因
    import random
    remaining = 50 - len(genes)
    for _ in range(remaining):
        genes.append(random.uniform(-2, 2))
    
    genome = Genome(genes=genes)
    
    print("Genome Script:")
    print(genome.decode())
    
    # 计算Discovery Score
    score = discovery_score.calculate(genome)
    
    print(f"\nDiscovery Score: {score['discovery_score']:.4f}")
    print(f"  - Train Accuracy: {score['train_accuracy']:.4f}")
    print(f"  - Test Accuracy: {score['test_accuracy']:.4f}")
    print(f"  - Simplicity: {score['simplicity']:.4f}")
    print(f"  - Generalization: {score['generalization']:.4f}")


def demo_curriculum_evolution():
    """演示Curriculum Evolution"""
    print("\n" + "="*60)
    print("  Curriculum Evolution Demo")
    print("="*60)
    
    inputs = TRAIN_INPUTS
    
    curriculum = CurriculumEvolution(inputs)
    
    print("Curriculum Stages:")
    for i, stage in enumerate(CurriculumEvolution.STAGES):
        print(f"  Stage {i+1}: {stage['description']}")
        print(f"    - Threshold: {stage['threshold']}")
        print(f"    - Generations: {stage['generations']}")
    
    print("\nRunning Curriculum Evolution...")
    
    # 运行完整课程
    best_genome, history = curriculum.run_full_curriculum()
    
    print("\n" + "="*60)
    print("  Curriculum Evolution Results")
    print("="*60)
    
    for stage_record in history:
        print(f"\n{stage_record['stage']}:")
        print(f"  Best Fitness: {stage_record['best_fitness']:.4f}")
    
    print(f"\nFinal Best Script:")
    print(best_genome.decode())


def demo_advanced_evolution():
    """演示高级进化引擎"""
    print("\n" + "="*60)
    print("  Advanced Evolution Engine Demo")
    print("="*60)
    
    target_fn = lambda x: x * 2 + 1
    
    # 创建高级进化引擎
    engine = AdvancedEvolutionEngine(TRAIN_INPUTS, TEST_INPUTS, target_fn)
    
    # 创建种子基因
    import random
    genes = [9.0, 100.0, 9.0, 100.0, 2.0, 1.0, 1.0, 2.0, 8.0, 100.0]
    remaining = 50 - len(genes)
    for _ in range(remaining):
        genes.append(random.uniform(-2, 2))
    
    seed_genome = Genome(genes=genes)
    
    print("Seed Genome:")
    print(seed_genome.decode())
    
    # 运行高级进化
    best_genome, discovery_score, history = engine.evolve_with_discovery(
        seed_genome,
        generations=200,
        pop_size=80
    )
    
    return best_genome, discovery_score


def demo_parallel_evaluation():
    """演示并行评估"""
    print("\n" + "="*60)
    print("  Parallel Evaluation Demo")
    print("="*60)
    
    target_fn = lambda x: x * 2 + 1
    
    # 创建种群
    import random
    population = []
    for _ in range(100):
        genes = []
        for _ in range(50):
            rand = random.random()
            if rand < 0.3:
                genes.append(float(random.randint(1, 20)))
            elif rand < 0.4:
                genes.append(float(random.randint(100, 110)))
            else:
                genes.append(random.uniform(-3, 3))
        population.append(Genome(genes=genes))
    
    print(f"Population size: {len(population)}")
    
    # 创建并行评估器
    evaluator = ParallelEvaluator()
    
    print(f"Using {evaluator.n_workers} workers for parallel evaluation")
    
    # 并行评估
    import time
    start_time = time.time()
    results = evaluator.evaluate_population(population, target_fn, TRAIN_INPUTS)
    elapsed = time.time() - start_time
    
    print(f"Evaluation completed in {elapsed:.2f}s")
    
    # 统计结果
    valid_count = sum(1 for r in results if r['valid_count'] > 0)
    print(f"Valid genomes: {valid_count}")
    
    # 最佳结果
    best_result = max(results, key=lambda x: x['fitness'])
    print(f"\nBest Fitness: {best_result['fitness']:.4f}")
    print(f"Best Error: {best_result['error']:.4f}")
    print(f"Best Script:\n{best_result['genome'].decode()}")


def main():
    """运行所有演示"""
    print("\n" + "="*70)
    print("  NeoGlyph Advanced Evolution Features Demo")
    print("="*70)
    
    # 1. Invalid Program Filter
    demo_invalid_program_filter()
    
    # 2. Discovery Score
    demo_discovery_score()
    
    # 3. Parallel Evaluation
    demo_parallel_evaluation()
    
    # 4. Curriculum Evolution (可选，耗时较长)
    print("\n运行Curriculum Evolution需要较长时间...")
    print("是否运行完整Curriculum Evolution? (y/n): ")
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--curriculum':
        demo_curriculum_evolution()
    
    # 5. Advanced Evolution (可选，耗时较长)
    if len(sys.argv) > 1 and sys.argv[1] == '--advanced':
        demo_advanced_evolution()
    
    print("\n" + "="*70)
    print("  Demo Complete")
    print("="*70)
    
    print("\n💡 Usage Tips:")
    print("  - Run with --curriculum flag to see Curriculum Evolution")
    print("  - Run with --advanced flag to see Advanced Evolution")
    print("  - Use ParallelEvaluator for large populations")
    print("  - Use InvalidProgramFilter to speed up evolution")
    print("  - Use DiscoveryScore to evaluate program quality")


if __name__ == '__main__':
    main()