#!/usr/bin/env python3
"""
Evolution Core 3.0 Demo

演示Tree Genome从随机结构进化到可读程序 y = 2*x + 1

目标：
- 随机初始化Tree Genome
- 自动发现 y = 2*x + 1
- 输出可读程序：2*x+1
- 输出VM代码
"""

import numpy as np
from neoglyph import TreeGenome, ArchiveMemory


# 测试数据
TRAIN_INPUTS = np.array([-5, -3, -1, 1, 3, 5], dtype=np.float32)


class TreeEvolutionEngine:
    """Tree Genome进化引擎"""
    
    def __init__(self, target_fn, inputs, pop_size=50, archive_size=20):
        self.target_fn = target_fn
        self.inputs = inputs
        self.pop_size = pop_size
        self.archive = ArchiveMemory(max_size=archive_size)
        self.population = []
        self.best_genome = None
        self.best_fitness = 0.0
        self.generation = 0
    
    def initialize_population(self, max_depth=3):
        """初始化种群"""
        self.population = []
        for _ in range(self.pop_size):
            genome = TreeGenome.create_random(max_depth)
            self.population.append(genome)
    
    def evaluate_population(self):
        """评估整个种群"""
        evaluations = []
        
        for genome in self.population:
            # 调试：检查评估结果
            eval_result = genome.evaluate_with_target(self.inputs, self.target_fn)
            
            if eval_result['valid_count'] > 0:
                fitness = genome.calculate_fitness(self.inputs, self.target_fn)
                evaluations.append({
                    'genome': genome,
                    'fitness': fitness,
                    'mse': eval_result['mse']
                })
                
                # 添加到Archive
                if fitness > 0.05:
                    self.archive.add(genome)
        
        # 排序
        evaluations.sort(key=lambda x: x['fitness'], reverse=True)
        
        # 更新最佳
        if evaluations and evaluations[0]['fitness'] > self.best_fitness:
            self.best_fitness = evaluations[0]['fitness']
            self.best_genome = evaluations[0]['genome'].copy()
            
            # 保护高fitness节点
            if self.best_fitness > 0.5:
                self.best_genome.protect_high_fitness_nodes(threshold=0.7)
        
        return evaluations
    
    def evolve(self, generations=200, mutation_rate=0.3):
        """进化循环"""
        print(f"\n{'='*60}")
        print(f"  Tree Genome Evolution")
        print(f"  Target: y = 2*x + 1")
        print(f"{'='*60}")
        
        # 初始化
        self.initialize_population(max_depth=3)
        
        # 初始评估
        print("\nGeneration 0 (Initial):")
        evaluations = self.evaluate_population()
        print(f"Best Fitness: {self.best_fitness:.4f}")
        
        if self.best_genome:
            print(f"Best Expression: {self.best_genome.to_expression()}")
            print(f"Best Complexity: {self.best_genome.get_complexity()}")
        else:
            print("No valid genome found initially")
        
        print(f"Archive Size: {len(self.archive.archive)}")
        
        # 进化循环
        for gen in range(1, generations + 1):
            self.generation = gen
            
            # 评估
            evaluations = self.evaluate_population()
            
            # 进度报告
            if gen % 20 == 0:
                print(f"\nGeneration {gen}:")
                print(f"Best Fitness: {self.best_fitness:.4f}")
                
                if self.best_genome:
                    print(f"Best Expression: {self.best_genome.to_expression()}")
                    print(f"Best MDL: {self.best_genome.mdl_score:.2f}")
                
                print(f"Archive Size: {len(self.archive.archive)}")
                
                # Archive统计
                stats = self.archive.get_statistics()
                print(f"Archive Avg Fitness: {stats['avg_fitness']:.4f}")
            
            # 生成下一代
            self.population = self._generate_next_generation(evaluations)
        
        # 最终结果
        print(f"\n{'='*60}")
        print(f"  Evolution Complete")
        print(f"{'='*60}")
        
        # 从Archive获取最佳
        archive_best = self.archive.get_best()
        if archive_best and archive_best.fitness > self.best_fitness:
            self.best_genome = archive_best
        
        if self.best_genome:
            print(f"\n✅ Best Expression: {self.best_genome.to_expression()}")
            print(f"Fitness: {self.best_genome.fitness:.4f}")
            print(f"MDL Score: {self.best_genome.mdl_score:.2f}")
            print(f"Complexity: {self.best_genome.get_complexity()}")
            
            print(f"\n📝 VM Code:")
            print(self.best_genome.to_vm_code())
            
            # 验证预测
            print(f"\n📊 Predictions:")
            print(f"{'Input':>6} {'Prediction':>12} {'Target':>10} {'Error':>8}")
            print("-" * 40)
            
            for x in TRAIN_INPUTS:
                pred = self.best_genome.root.evaluate(x)
                target = self.target_fn(x)
                error = abs(pred - target)
                print(f"{x:>6.0f} {pred:>12.4f} {target:>10.0f} {error:>8.4f}")
        else:
            print("\n⚠️  No valid genome found during evolution")
            print("This might indicate issues with:")
            print("  - Random tree generation")
            print("  - Evaluation function")
            print("  - Target function matching")
        
        # Archive内容
        print(f"\n📚 Archive Contents:")
        stats = self.archive.get_statistics()
        print(f"Total Programs: {stats['size']}")
        print(f"Best Fitness: {stats['best_fitness']:.4f}")
        print(f"Avg MDL: {stats['avg_mdl']:.2f}")
        
        return self.best_genome
        
        # Archive内容
        print(f"\n📚 Archive Contents:")
        stats = self.archive.get_statistics()
        print(f"Total Programs: {stats['size']}")
        print(f"Best Fitness: {stats['best_fitness']:.4f}")
        print(f"Avg MDL: {stats['avg_mdl']:.2f}")
        
        return self.best_genome
    
    def _generate_next_generation(self, evaluations):
        """生成下一代"""
        import random
        
        new_population = []
        
        # 精英保留
        elite_count = int(self.pop_size * 0.2)
        for e in evaluations[:elite_count]:
            new_population.append(e['genome'].copy())
        
        # 从Archive添加
        archive_best = self.archive.get_best()
        if archive_best:
            new_population.append(archive_best.copy())
        
        # 生成新个体
        pool = [e['genome'] for e in evaluations[:int(self.pop_size * 0.5)]]
        
        while len(new_population) < self.pop_size:
            if len(pool) >= 2:
                parent1 = random.choice(pool)
                parent2 = random.choice(pool)
            else:
                parent1 = self.best_genome.copy()
                parent2 = TreeGenome.create_random(2)
            
            # Crossover（简化）
            child = parent1.copy()
            
            # Mutation
            child.mutate(
                mutation_rate=0.3,
                fitness=max(parent1.fitness, parent2.fitness)
            )
            
            new_population.append(child)
        
        return new_population[:self.pop_size]


def main():
    """运行Evolution Core 3.0 Demo"""
    print("\n" + "="*70)
    print("  Evolution Core 3.0")
    print("  Tree Genome: From Random Code to Readable Program")
    print("="*70)
    
    # 目标函数
    target_fn = lambda x: x * 2 + 1
    
    print("\n🎯 Target: y = 2*x + 1")
    print("Challenge: Evolve from random tree structure to this program")
    
    # 创建进化引擎
    engine = TreeEvolutionEngine(
        target_fn=target_fn,
        inputs=TRAIN_INPUTS,
        pop_size=50,
        archive_size=20
    )
    
    # 运行进化
    best_genome = engine.evolve(generations=200, mutation_rate=0.3)
    
    # 成功标准
    print(f"\n{'='*60}")
    print(f"  Success Criteria")
    print(f"{'='*60}")
    
    # 检查是否接近目标
    target_expr = "2*x + 1"
    best_expr = best_genome.to_expression()
    
    print(f"Target Expression: {target_expr}")
    print(f"Best Expression: {best_expr}")
    
    # 计算最终误差
    total_error = 0
    for x in TRAIN_INPUTS:
        pred = best_genome.root.evaluate(x)
        target = target_fn(x)
        total_error += abs(pred - target)
    
    avg_error = total_error / len(TRAIN_INPUTS)
    
    if avg_error < 0.5:
        print(f"\n🎉 SUCCESS! Average error: {avg_error:.4f}")
        print("NeoGlyph discovered the target program structure!")
    elif avg_error < 2.0:
        print(f"\n📈 Good progress! Average error: {avg_error:.4f}")
        print("Close to target, but not exact match.")
    else:
        print(f"\n🔄 Need more evolution. Average error: {avg_error:.4f}")
        print("Try more generations or larger population.")
    
    return best_genome


if __name__ == '__main__':
    best_genome = main()