"""
NeoGlyph v4 Evolution Engine - Unified API
===========================================

单一 EvolutionEngine，支持 linear 和 tree 两种 genome 类型。
所有高级功能（并行评估、Discovery Score、Curriculum Learning 等）
通过可配置策略实现，不再创建新的 Engine 类。
"""

import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from .genome import Genome, TreeGenome


# ============================================================================
# 可配置策略
# ============================================================================

class Evaluator:
    """评估器基类 - 可配置策略"""

    def evaluate(self, genome, inputs, target_fn):
        raise NotImplementedError


class SequentialEvaluator(Evaluator):
    """串行评估器（默认）"""

    def evaluate(self, genome, inputs, target_fn):
        return genome.calculate_fitness(inputs, target_fn)


class ParallelEvaluator(Evaluator):
    """并行评估器"""

    def __init__(self, n_workers=None):
        import multiprocessing as mp
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)

    def evaluate(self, genome, inputs, target_fn):
        return genome.calculate_fitness(inputs, target_fn)

    def evaluate_all(self, population, inputs, target_fn):
        def _eval(g):
            g.calculate_fitness(inputs, target_fn)
            return g.fitness

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            fitnesses = list(executor.map(_eval, population))

        for i, f in enumerate(fitnesses):
            population[i].fitness = f


class Selector:
    """选择器基类 - 可配置策略"""

    def select(self, population, k=1):
        raise NotImplementedError


class TournamentSelector(Selector):
    """锦标赛选择"""

    def __init__(self, tournament_size=3):
        self.tournament_size = tournament_size

    def select(self, population, k=1):
        if k == 1:
            candidates = random.sample(
                population,
                min(self.tournament_size, len(population))
            )
            return max(candidates, key=lambda g: g.fitness)

        selected = []
        for _ in range(k):
            candidates = random.sample(
                population,
                min(self.tournament_size, len(population))
            )
            selected.append(max(candidates, key=lambda g: g.fitness))
        return selected


class Scorer:
    """评分器基类 - 可配置策略"""

    def compute_score(self, genome, accuracy, complexity):
        raise NotImplementedError


class FitnessScorer(Scorer):
    """标准 Fitness 评分器"""

    def __init__(self, mdl_weight=0.02):
        self.mdl_weight = mdl_weight

    def compute_score(self, genome, accuracy, complexity):
        mdl_penalty = (complexity / 30.0) * self.mdl_weight
        return max(accuracy - mdl_penalty, 0.001)


class DiscoveryScorer(Scorer):
    """Discovery Score 评分器

    三维度评分：
    - accuracy: 准确度
    - simplicity: 简洁性
    - generalization: 泛化能力
    """

    def __init__(self, test_inputs=None, test_y=None, mdl_weight=0.02):
        self.test_inputs = test_inputs
        self.test_y = test_y
        self.mdl_weight = mdl_weight

    def compute_score(self, genome, accuracy, complexity):
        train_score = accuracy

        # 泛化能力
        generalization = 0.0
        if self.test_inputs is not None and self.test_y is not None:
            try:
                result = genome.evaluate_array(self.test_inputs, self.test_y)
                generalization = result['accuracy']
            except Exception:
                generalization = 0.0

        # 简洁性
        simplicity = 1.0 / (1.0 + complexity / 10.0)

        discovery_score = (
            train_score * 0.4 +
            generalization * 0.3 +
            simplicity * 0.3
        )
        return discovery_score


class EarlyStopper:
    """早停策略

    监控最佳 fitness 是否持续不提升，若超过 patience 代无改进则停止。

    Parameters
    ----------
    patience : int
        容忍无改进的最大代数
    min_delta : float
        视为"改进"的最小 fitness 提升量
    """

    def __init__(self, patience=20, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_fitness = 0.0
        self.counter = 0

    def reset(self):
        self.best_fitness = 0.0
        self.counter = 0

    def should_stop(self, current_fitness):
        if current_fitness > self.best_fitness + self.min_delta:
            self.best_fitness = current_fitness
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ============================================================================
# 统一 EvolutionEngine
# ============================================================================

class EvolutionEngine:
    """统一进化引擎 - NeoGlyph v4

    支持 genome_type="linear"（向后兼容）和 genome_type="tree"（推荐）。

    所有高级功能通过可配置策略实现：
        evaluator: 评估策略（Sequential/Parallel）
        scorer: 评分策略（Fitness/Discovery）
        selector: 选择策略（Tournament）

    Parameters
    ----------
    genome_type : str
        "linear" 或 "tree"
    pop_size : int
        种群大小
    max_depth : int
        表达式树最大深度（仅 tree 模式）
    mutation_rate : float
        变异率
    elite_ratio : float
        精英保留比例
    evaluator : Evaluator or None
        评估策略，None 则使用 SequentialEvaluator
    scorer : Scorer or None
        评分策略，None 则使用 FitnessScorer
    selector : Selector or None
        选择策略，None 则使用 TournamentSelector
    parallel : bool
        是否启用并行评估（覆盖 evaluator）
    archive : bool
        是否启用 Archive Memory（预留）
    verbose : bool
        是否打印进化过程
    random_state : int or None
        随机种子
    """

    def __init__(self, genome_type="linear", pop_size=50, max_depth=3,
                 mutation_rate=0.3, elite_ratio=0.1,
                 evaluator=None, scorer=None, selector=None,
                 early_stopper=None,
                 fitness_sharing=None,
                 parallel=False, archive=False, verbose=False,
                 random_state=None,
                 variable_names=None,
                 # 向后兼容参数
                 initial_genome=None):
        self.genome_type = genome_type
        self.pop_size = pop_size
        self.max_depth = max_depth
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.verbose = verbose
        self.archive_enabled = archive
        self.variable_names = variable_names or ['x']

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        # 策略
        if parallel and evaluator is None:
            self.evaluator = ParallelEvaluator()
        else:
            self.evaluator = evaluator or SequentialEvaluator()
        self.scorer = scorer or FitnessScorer()
        self.selector = selector or TournamentSelector()
        self.early_stopper = early_stopper
        self.fitness_sharing = fitness_sharing  # FitnessSharing 实例，None 表示不启用

        # 状态
        self.generation = 0
        self.population = []
        self.best_genome = None
        self.best_fitness = 0.0
        self.history = []
        self.archive = {} if archive else None

        # 初始化种群
        if genome_type == "tree":
            self._initialize_tree_population()
        else:
            self._initialize_linear_population(initial_genome)

    # ---- 种群初始化 ----

    def _initialize_tree_population(self):
        self.population = [
            TreeGenome.create_random(max_depth=self.max_depth, variable_names=self.variable_names)
            for _ in range(self.pop_size)
        ]

    def _initialize_linear_population(self, initial_genome=None):
        if initial_genome is not None:
            self.population = [initial_genome] + [
                Genome(genes=initial_genome.genes.copy())
                for _ in range(self.pop_size - 1)
            ]
        else:
            self.population = [Genome(length=20) for _ in range(self.pop_size)]

    # ---- 向后兼容 API（linear 模式）----

    def initialize_population(self):
        """重新初始化种群（向后兼容 TreeEvolutionEngine）"""
        if self.genome_type == "tree":
            self._initialize_tree_population()
        else:
            self._initialize_linear_population()

    def evaluate_population(self, inputs, target_fn):
        """评估整个种群（向后兼容 TreeEvolutionEngine）"""
        for genome in self.population:
            genome.calculate_fitness(inputs, target_fn)

    @staticmethod
    def selection(population, tournament_size=3):
        """锦标赛选择（向后兼容 TreeEvolutionEngine 静态方法）"""
        candidates = random.sample(population, min(tournament_size, len(population)))
        return max(candidates, key=lambda g: g.fitness)

    def evaluate_genome(self, genome, target_fn=None, input_vars=None):
        """评估单个线性 Genome（向后兼容）"""
        from .vm import NeoGlyphVM
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
                'fitness': 0.0, 'accuracy': 0.0,
                'speed': 0.0, 'memory': 0.0,
                'instruction': 0.0, 'error': 0.0,
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
                if hasattr(selected_genome, 'genes') and hasattr(eval_item['genome'], 'genes'):
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
        """生成下一代（linear 模式，向后兼容）"""
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

    def print_report(self, report):
        improvement_str = (
            f"+{report['improvement']:.1f}%" if report['improvement'] > 0 else
            f"{report['improvement']:.1f}%" if report['improvement'] < 0 else "0%"
        )
        print(f"Generation {report['generation']}:")
        print(f"  Population: {report['population']}")
        print(f"  Best Fitness: {report['best_fitness']:.4f}")
        print(f"  Improvement: {improvement_str}")
        if hasattr(report['best_genome'], 'genes'):
            print(f"  Best Genome: {len(report['best_genome'].genes)} genes")
            print(f"  Decoded Script:")
            print(report['best_genome'].decode())
        else:
            print(f"  Best Expression: {report['best_genome'].to_expression()}")
        print()

    def get_summary(self):
        if not self.history:
            return None

        summary = {
            'total_generations': self.generation,
            'initial_fitness': self.history[0]['best_fitness'],
            'final_fitness': self.best_fitness,
            'best_genome': self.best_genome,
            'history': [
                {'generation': r['generation'], 'fitness': r['best_fitness']}
                for r in self.history
            ]
        }

        if self.genome_type == "tree" and self.best_genome:
            summary['best_expression'] = self.best_genome.to_expression()
        elif self.genome_type == "linear":
            summary['average_improvement'] = (
                sum(r['improvement'] for r in self.history) / len(self.history)
            )

        return summary

    # ---- 统一 evolve 接口 ----

    def evolve(self, inputs=None, target_fn=None, generations=100,
               input_vars=None, verbose=None):
        """运行进化

        两种模式：
        - tree 模式: evolve(inputs=X, target_fn=lambda x: y)
        - linear 模式: evolve(target_fn=fn, generations=N, verbose=True)

        返回：best_genome (tree) 或 reports (linear)
        """
        if verbose is None:
            verbose = self.verbose

        if self.genome_type == "tree":
            return self._evolve_tree(inputs, target_fn, generations, verbose)
        else:
            return self._evolve_linear(target_fn, input_vars, generations, verbose)

    def _evolve_tree(self, inputs, target_fn, generations, verbose):
        """Tree 模式进化"""
        if not self.population:
            self._initialize_tree_population()

        if self.early_stopper:
            self.early_stopper.reset()

        for gen in range(generations):
            self.generation = gen + 1

            self._evaluate_tree_population(inputs, target_fn)

            current_best = max(self.population, key=lambda g: g.fitness)
            if current_best.fitness > self.best_fitness:
                self.best_fitness = current_best.fitness
                self.best_genome = current_best.copy()

            avg_fitness = float(np.mean([g.fitness for g in self.population]))
            self.history.append({
                'generation': self.generation,
                'best_fitness': self.best_fitness,
                'avg_fitness': avg_fitness,
            })

            if self.early_stopper and self.early_stopper.should_stop(self.best_fitness):
                if verbose:
                    print(f"Early stopping at Gen {gen} (no improvement for "
                          f"{self.early_stopper.patience} generations)")
                break

            if verbose and gen % 10 == 0:
                expr = self.best_genome.to_expression() if self.best_genome else 'N/A'
                print(f"Gen {gen:3d}: Best={self.best_fitness:.4f}, "
                      f"Avg={avg_fitness:.4f}, Expr={expr}")

            self._generate_tree_next_generation()

        return self.best_genome

    def _evaluate_tree_population(self, inputs, target_fn):
        """评估 tree 种群，支持适应度共享"""
        if isinstance(self.evaluator, ParallelEvaluator):
            self.evaluator.evaluate_all(self.population, inputs, target_fn)
        else:
            for genome in self.population:
                self.evaluator.evaluate(genome, inputs, target_fn)
        
        # 适应度共享：惩罚相似个体，鼓励多样性
        if self.fitness_sharing is not None:
            adjusted = self.fitness_sharing.apply(self.population)
            for i, g in enumerate(self.population):
                g._raw_fitness = g.fitness  # 保留原始 fitness
                g.fitness = adjusted[i]

    def _generate_tree_next_generation(self):
        """生成下一代（tree 模式）"""
        sorted_pop = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        elite_count = max(1, int(self.pop_size * self.elite_ratio))
        new_pop = [g.copy() for g in sorted_pop[:elite_count]]

        while len(new_pop) < self.pop_size:
            if hasattr(self.selector, 'select') and callable(self.selector.select):
                parents = self.selector.select(sorted_pop, k=2)
                if len(parents) >= 2:
                    p1, p2 = parents[0], parents[1]
                else:
                    p1 = p2 = parents[0]
            else:
                p1 = self.selector.select(sorted_pop)
                p2 = self.selector.select(sorted_pop)

            child = TreeGenome.crossover(p1, p2)
            child.mutate(self.mutation_rate, fitness=p1.fitness)
            new_pop.append(child)

        self.population = new_pop[:self.pop_size]

    def _evolve_linear(self, target_fn, input_vars, generations, verbose):
        """Linear 模式进化（向后兼容）"""
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
                improvement = (
                    (best_eval['fitness'] - self.best_fitness) / self.best_fitness * 100
                )

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


# ============================================================================
# 向后兼容别名
# ============================================================================

def TreeEvolutionEngine(pop_size=50, max_depth=3, mutation_rate=0.3,
                        elite_ratio=0.1):
    """TreeEvolutionEngine 向后兼容别名

    直接返回 EvolutionEngine(genome_type="tree", ...)
    """
    return EvolutionEngine(
        genome_type="tree",
        pop_size=pop_size,
        max_depth=max_depth,
        mutation_rate=mutation_rate,
        elite_ratio=elite_ratio,
    )


# ============================================================================
# 高级进化策略
# ============================================================================

class FitnessSharing:
    """适应度共享 - 防止早熟收敛
    
    惩罚过于相似的个体，鼓励种群多样性。
    相似度基于表达式结构重叠。
    """
    
    def __init__(self, sigma=0.3, alpha=1.0):
        self.sigma = sigma
        self.alpha = alpha
    
    def apply(self, population):
        """应用适应度共享，返回调整后的 fitness 列表"""
        n = len(population)
        if n <= 1:
            return [g.fitness for g in population]
        
        expressions = []
        for g in population:
            try:
                expressions.append(g.to_expression())
            except Exception:
                expressions.append("")
        
        adjusted = []
        for i, g in enumerate(population):
            sharing_sum = 0.0
            for j in range(n):
                if i == j:
                    continue
                d = self._expression_distance(expressions[i], expressions[j])
                if d < self.sigma:
                    sharing_sum += 1.0 - (d / self.sigma) ** self.alpha
            adjusted.append(g.fitness / (1.0 + sharing_sum))
        return adjusted
    
    def _expression_distance(self, expr1, expr2):
        """计算表达式结构距离 (0=完全相同, 1=完全不同)"""
        if not expr1 or not expr2:
            return 1.0
        ops1 = set(c for c in expr1 if c in '+-*/()')
        ops2 = set(c for c in expr2 if c in '+-*/()')
        if not ops1 and not ops2:
            return 0.0 if expr1 == expr2 else 1.0
        overlap = len(ops1 & ops2)
        total = len(ops1 | ops2)
        return 1.0 - (overlap / max(total, 1))


class IslandModel:
    """岛模型 - 保持种群多样性
    
    将种群分成多个岛，各自独立进化，定期迁移个体。
    
    Parameters
    ----------
    n_islands : int
        岛的数量
    pop_size : int
        每个岛的种群大小
    migration_interval : int
        迁移间隔（代数）
    migration_rate : float
        每次迁移的个体比例
    """
    
    def __init__(self, n_islands=4, pop_size=50, migration_interval=10, 
                 migration_rate=0.1):
        self.n_islands = n_islands
        self.pop_size = pop_size
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.islands = []
        self.best_genome = None
        self.best_fitness = 0.0
    
    def initialize(self, **engine_kwargs):
        """初始化所有岛
        
        Parameters
        ----------
        **engine_kwargs : 传递给 EvolutionEngine 的参数
            genome_type, max_depth, mutation_rate, elite_ratio, etc.
        """
        island_pop = max(5, self.pop_size // self.n_islands)
        self.islands = []
        for i in range(self.n_islands):
            kwargs = engine_kwargs.copy()
            kwargs['pop_size'] = island_pop
            kwargs['random_state'] = engine_kwargs.get('random_state', np.random.randint(0, 10000)) + i * 100
            engine = EvolutionEngine(**kwargs)
            self.islands.append({
                'engine': engine,
                'population': engine.population,
                'best_fitness': 0.0,
            })
    
    def _migrate(self):
        """在岛之间迁移个体"""
        if len(self.islands) < 2:
            return
        for i, island in enumerate(self.islands):
            pop = island['population']
            if not pop or len(pop) < 2:
                continue
            n_migrate = max(1, int(len(pop) * self.migration_rate))
            migrants = sorted(pop, key=lambda g: g.fitness, reverse=True)[:n_migrate]
            target = self.islands[(i + 1) % len(self.islands)]
            target_pop = target['population']
            if target_pop and len(target_pop) >= n_migrate:
                target_pop.sort(key=lambda g: g.fitness)
                for j in range(n_migrate):
                    target_pop[j] = migrants[j].copy()
    
    def evolve(self, inputs, target_fn, generations=100, verbose=False):
        """运行岛模型进化（tree 模式）
        
        Parameters
        ----------
        inputs : array-like
            输入数据
        target_fn : callable or array-like
            目标函数或目标值数组
        generations : int
            总进化代数
        verbose : bool
            是否打印进度
        
        Returns
        -------
        result : dict
            包含 best_fitness, best_genome, islands
        """
        if not self.islands:
            raise ValueError("请先调用 initialize()")
        
        island_gens = max(1, self.migration_interval)
        n_cycles = max(1, generations // self.migration_interval)
        
        for cycle in range(n_cycles):
            for i, island in enumerate(self.islands):
                engine = island['engine']
                # 使用统一的 evolve 接口
                engine.evolve(inputs, target_fn, generations=island_gens, verbose=False)
                island['population'] = engine.population
                island['best_fitness'] = engine.best_fitness
                if engine.best_fitness > self.best_fitness:
                    self.best_fitness = engine.best_fitness
                    self.best_genome = engine.best_genome.copy() if engine.best_genome else None
            
            self._migrate()
            
            if verbose and cycle % max(1, n_cycles // 10) == 0:
                fits = [island['best_fitness'] for island in self.islands]
                print(f"  Cycle {cycle}/{n_cycles}: best={max(fits):.4f}, "
                      f"islands={[f'{f:.3f}' for f in fits]}")
        
        return {
            'best_fitness': self.best_fitness,
            'best_genome': self.best_genome,
            'islands': self.islands,
        }