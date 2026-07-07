"""
NeoGlyph v4 Application Layer - 统一应用 API
=============================================

所有应用统一继承 BaseApplication。
统一接口：fit / predict / score / expression / save / load
"""

import json
import pickle
import numpy as np
import random
from .genome import TreeGenome
from .evolution import EvolutionEngine, SequentialEvaluator, ParallelEvaluator


class BaseApplication:
    """NeoGlyph 应用基类

    所有应用必须继承此类，实现统一的接口。
    子类只需实现 _build_engine() 和 _postprocess_result()。
    """

    def __init__(self, random_state=None):
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        self.best_genome_ = None
        self.history_ = []
        self._engine = None

    def _build_engine(self):
        """子类重写：构建 EvolutionEngine 实例"""
        raise NotImplementedError

    def fit(self, X, y):
        """拟合模型

        Parameters
        ----------
        X : array-like
            输入数据
        y : array-like
            目标值

        Returns
        -------
        self : BaseApplication
        """
        X = np.asarray(X, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()

        self._engine = self._build_engine()
        self._engine.best_genome = None
        self._engine.best_fitness = 0.0

        best_fitness = 0.0
        best_genome = None

        for gen in range(self._engine.generations if hasattr(self._engine, 'generations') else 100):
            self._engine.generation = gen + 1

            self._engine._evaluate_tree_population(X, y)

            current_best = max(self._engine.population, key=lambda g: g.fitness)
            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                best_genome = current_best.copy()

            avg_fitness = float(np.mean([g.fitness for g in self._engine.population]))
            self.history_.append({
                'generation': gen + 1,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
            })

            self._engine._generate_tree_next_generation()

        self.best_genome_ = best_genome
        return self

    def predict(self, X):
        """使用最佳表达式预测
        
        支持单变量和多变量：
        - 单变量: X = [1.0, 2.0, 3.0]
        - 多变量: X = [[1.0, 2.0], [3.0, 4.0]]  每行对应多个变量
        """
        if self.best_genome_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.ravel()
        predictions = np.zeros(len(X))

        var_names = sorted(self.best_genome_._collect_variables(self.best_genome_.root))
        if not var_names:
            var_names = ['x']

        for i in range(len(X)):
            try:
                if X.ndim == 1:
                    predictions[i] = float(self.best_genome_.root.evaluate(X[i]))
                else:
                    x_dict = {var_names[j]: float(X[i, j]) for j in range(min(X.shape[1], len(var_names)))}
                    predictions[i] = float(self.best_genome_.root.evaluate(x_dict))
            except Exception:
                predictions[i] = 0.0

        return predictions

    def score(self, X, y):
        """计算 R² 分数"""
        y_pred = self.predict(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1 - ss_res / ss_tot

    def expression(self):
        """获取最佳表达式（简化后）"""
        if self.best_genome_ is None:
            return "N/A"
        return self.best_genome_.simplify().to_expression()

    def best_expression(self):
        """向后兼容别名"""
        return self.expression()

    def save(self, filepath):
        """保存模型到文件"""
        data = {
            'genome': self.best_genome_,
            'history': self.history_,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath):
        """从文件加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.best_genome_ = data['genome']
        self.history_ = data['history']
        return self


class SymbolicRegressor(BaseApplication):
    """符号回归器

    从输入输出数据中自动发现数学表达式。
    提供与 scikit-learn 兼容的 fit/predict/score 接口。

    Parameters
    ----------
    pop_size : int, 种群大小
    max_depth : int, 表达式树最大深度
    generations : int, 进化代数
    mutation_rate : float, 变异率
    elite_ratio : float, 精英保留比例
    mdl_weight : float, MDL 复杂度惩罚权重
    parallel : bool, 是否启用并行评估
    verbose : bool, 是否打印进化过程
    random_state : int or None, 随机种子
    """

    def __init__(self, pop_size=100, max_depth=4, generations=200,
                 mutation_rate=0.3, elite_ratio=0.1, mdl_weight=0.02,
                 parallel=False, verbose=False, random_state=None):
        super().__init__(random_state=random_state)
        self.pop_size = pop_size
        self.max_depth = max_depth
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.mdl_weight = mdl_weight
        self.parallel = parallel
        self.verbose = verbose

    def _build_engine(self, variable_names=None):
        evaluator = ParallelEvaluator() if self.parallel else SequentialEvaluator()
        return EvolutionEngine(
            genome_type="tree",
            pop_size=self.pop_size,
            max_depth=self.max_depth,
            mutation_rate=self.mutation_rate,
            elite_ratio=self.elite_ratio,
            evaluator=evaluator,
            verbose=self.verbose,
            variable_names=variable_names,
        )

    def _compute_fitness(self, genome, X, y):
        result = genome.evaluate_array(X, y)
        accuracy = result['accuracy']
        complexity = genome.get_complexity()
        mdl_penalty = (complexity / 30.0) * self.mdl_weight
        genome.fitness = max(accuracy - mdl_penalty, 0.001)
        genome.mdl_score = result['mse'] + complexity * 0.01
        return genome.fitness

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        # 保持 X 的原始形状（1D 单变量，2D 多变量）

        # 自动检测变量名
        if X.ndim == 2 and X.shape[1] > 1:
            variable_names = [chr(ord('x') + i) if i < 26 else f'v{i}' for i in range(X.shape[1])]
        else:
            variable_names = None

        self._engine = self._build_engine(variable_names=variable_names)
        best_fitness = 0.0
        best_genome = None

        for gen in range(self.generations):
            self._engine.generation = gen + 1

            for genome in self._engine.population:
                self._compute_fitness(genome, X, y)

            current_best = max(self._engine.population, key=lambda g: g.fitness)
            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                best_genome = current_best.copy()

            avg_fitness = float(np.mean([g.fitness for g in self._engine.population]))
            self.history_.append({
                'generation': gen + 1,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
            })

            if self.verbose and gen % 10 == 0:
                expr = best_genome.simplify().to_expression() if best_genome else 'N/A'
                print(f"Gen {gen:4d}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")
                print(f"         Expr: {expr}")

            self._engine._generate_tree_next_generation()

        self.best_genome_ = best_genome
        return self


class PhysicsDiscoverer(BaseApplication):
    """物理公式发现器

    从实验数据中自动发现物理定律。
    """

    def __init__(self, pop_size=150, max_depth=5, generations=300,
                 mdl_weight=0.01, verbose=False, random_state=None):
        super().__init__(random_state=random_state)
        self._regressor = SymbolicRegressor(
            pop_size=pop_size,
            max_depth=max_depth,
            generations=generations,
            mdl_weight=mdl_weight,
            verbose=verbose,
            random_state=random_state,
        )

    @property
    def regressor(self):
        """向后兼容：暴露内部 SymbolicRegressor"""
        return self._regressor

    def _build_engine(self):
        return self._regressor._build_engine()

    def fit(self, X, y):
        self._regressor.fit(X, y)
        self.best_genome_ = self._regressor.best_genome_
        self.history_ = self._regressor.history_
        return self

    def discover(self, X, y, variable_name='x'):
        """从数据中发现物理公式

        Returns
        -------
        result : dict
            包含 expression, r2, error 等信息
        """
        self.fit(X, y)

        expr = self.expression()
        r2 = self.score(X, y)

        y_pred = self.predict(X)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        mse = float(np.mean((y_arr - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_arr - y_pred)))

        return {
            'expression': expr,
            'r2_score': r2,
            'mse': mse,
            'mae': mae,
            'complexity': self.best_genome_.get_complexity() if self.best_genome_ else 0,
            'history': self.history_,
            'genome': self.best_genome_,
        }


# 向后兼容别名
def ParallelSymbolicRegressor(*args, **kwargs):
    """ParallelSymbolicRegressor 向后兼容别名"""
    kwargs['parallel'] = True
    return SymbolicRegressor(*args, **kwargs)