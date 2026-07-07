# API 参考

## 应用层

### `SymbolicRegressor`

符号回归器，从数据中自动发现数学表达式。sklearn 风格接口。

```python
from neoglyph import SymbolicRegressor

reg = SymbolicRegressor(
    pop_size=100,           # 种群大小，默认 100
    max_depth=4,            # 表达式树最大深度，默认 4
    generations=200,        # 进化代数，默认 200
    mutation_rate=0.3,      # 变异率，默认 0.3
    elite_ratio=0.1,        # 精英保留比例，默认 0.1
    mdl_weight=0.02,        # MDL 复杂度惩罚权重，默认 0.02
    parallel=False,         # 是否启用并行评估，默认 False
    verbose=False,          # 是否打印进化过程，默认 False
    random_state=None,      # 随机种子，默认 None
)
```

**方法：**

| 方法 | 说明 |
|------|------|
| `fit(X, y)` | 拟合模型，返回 self |
| `predict(X)` | 用最佳表达式预测 |
| `score(X, y)` | 计算 R² 分数 |
| `expression()` | 获取最佳表达式（简化后） |
| `save(filepath)` | 保存模型到文件 |
| `load(filepath)` | 从文件加载模型 |

**属性：**

| 属性 | 说明 |
|------|------|
| `best_genome_` | 最佳 TreeGenome 对象 |
| `history_` | 进化历史列表，每项含 generation/best_fitness/avg_fitness |

### `PhysicsDiscoverer`

物理公式发现器，从实验数据中自动发现物理定律。

```python
from neoglyph import PhysicsDiscoverer

disc = PhysicsDiscoverer(
    pop_size=150,           # 种群大小，默认 150
    max_depth=5,            # 最大深度，默认 5
    generations=300,        # 进化代数，默认 300
    mdl_weight=0.01,        # MDL 惩罚权重，默认 0.01
    verbose=False,          # 是否打印进度
    random_state=None,      # 随机种子
)
```

**方法：**

| 方法 | 说明 |
|------|------|
| `discover(X, y)` | 发现公式，返回 dict（含 expression/r2_score/mse/mae/complexity/history/genome） |
| `fit(X, y)` | 拟合模型 |
| `predict(X)` | 预测 |
| `score(X, y)` | 计算 R² |
| `expression()` | 获取表达式 |

### `BaseApplication`

应用基类，所有应用继承此类。

```python
class BaseApplication:
    def fit(self, X, y): ...
    def predict(self, X): ...
    def score(self, X, y): ...
    def expression(self): ...
    def save(self, filepath): ...
    def load(self, filepath): ...
```

---

## 进化引擎

### `EvolutionEngine`

统一进化引擎，支持 linear 和 tree 两种基因组类型。

```python
from neoglyph import EvolutionEngine

engine = EvolutionEngine(
    genome_type="tree",         # "linear" 或 "tree"
    pop_size=50,                # 种群大小，默认 50
    max_depth=3,                # 最大深度（仅 tree 模式），默认 3
    mutation_rate=0.3,          # 变异率，默认 0.3
    elite_ratio=0.1,            # 精英保留比例，默认 0.1
    evaluator=None,             # Evaluator 策略，默认 SequentialEvaluator
    scorer=None,                # Scorer 策略，默认 FitnessScorer
    selector=None,              # Selector 策略，默认 TournamentSelector
    early_stopper=None,         # EarlyStopper 实例
    fitness_sharing=None,       # FitnessSharing 实例
    parallel=False,             # 是否启用并行评估
    archive=False,              # 是否启用 Archive Memory
    verbose=False,              # 是否打印进化过程
    random_state=None,          # 随机种子
    variable_names=None,        # 变量名列表，如 ['x', 'y']，默认 ['x']
)
```

**tree 模式关键方法：**

| 方法 | 说明 |
|------|------|
| `_initialize_tree_population()` | 初始化树种群 |
| `_evaluate_tree_population(inputs, target_fn)` | 评估整个种群 |
| `_generate_tree_next_generation()` | 生成下一代 |
| `evolve(inputs, target_fn, generations, verbose)` | 运行进化 |

### 可配置策略

#### `Evaluator` (基类)

```python
class Evaluator:
    def evaluate(self, genome, inputs, target_fn): ...
```

#### `SequentialEvaluator`

串行评估器，默认策略。

#### `ParallelEvaluator`

并行评估器，使用线程池。

```python
ParallelEvaluator(n_workers=None)  # None = CPU 核心数 - 1
```

#### `Selector` (基类)

```python
class Selector:
    def select(self, population, k=1): ...
```

#### `TournamentSelector`

锦标赛选择器。

```python
TournamentSelector(tournament_size=3)
```

#### `Scorer` (基类)

```python
class Scorer:
    def compute_score(self, genome, accuracy, complexity): ...
```

#### `FitnessScorer`

标准 Fitness 评分器。

```python
FitnessScorer(mdl_weight=0.02)
```

#### `DiscoveryScorer`

三维度评分：准确度 + 泛化 + 简洁性。

```python
DiscoveryScorer(test_inputs=None, test_y=None, mdl_weight=0.02)
```

### 高级进化策略

#### `EarlyStopper`

早停策略，监控 fitness 是否持续不提升。

```python
from neoglyph import EarlyStopper

stopper = EarlyStopper(
    patience=20,        # 容忍无改进的最大代数
    min_delta=0.0001,   # 视为"改进"的最小 fitness 提升
)

stopper.reset()
if stopper.should_stop(current_fitness):
    break  # 停止进化
```

#### `FitnessSharing`

适应度共享，惩罚相似个体，鼓励多样性。

```python
from neoglyph import FitnessSharing

sharing = FitnessSharing(
    sigma=0.3,      # 相似度阈值
    alpha=1.0,      # 惩罚强度
)

adjusted_fitnesses = sharing.apply(population)
```

#### `IslandModel`

岛模型，将种群分成多个岛独立进化，定期迁移。

```python
from neoglyph import IslandModel

model = IslandModel(
    n_islands=4,                # 岛的数量
    pop_size=50,                # 每个岛的种群大小
    migration_interval=10,      # 迁移间隔（代数）
    migration_rate=0.1,         # 每次迁移的个体比例
)

model.initialize(
    genome_type="tree",
    max_depth=3,
    mutation_rate=0.3,
    elite_ratio=0.1,
)

result = model.evolve(X, y, generations=200, verbose=True)
```

---

## 基因组

### `TreeGenome`

树状基因组，表示数学表达式。

```python
from neoglyph import TreeGenome

tree = TreeGenome(root_node)  # root_node 是 OperationNode/VariableNode/ConstantNode
```

**方法：**

| 方法 | 说明 |
|------|------|
| `create_random(max_depth, variable_names)` | 静态方法，创建随机树 |
| `to_expression()` | 转换为可读表达式字符串 |
| `to_vm_code()` | 转换为 VM 指令代码 |
| `simplify()` | 符号简化，返回简化后的 TreeGenome |
| `evaluate(inputs)` | 在多个输入上评估 |
| `evaluate_with_target(inputs, target_fn)` | 带目标函数的评估，返回 dict |
| `evaluate_array(X, y)` | 数组评估（向量化），返回 dict |
| `calculate_fitness(inputs, target_fn, mdl_weight)` | 计算 fitness |
| `get_complexity()` | 获取表达式复杂度（节点数） |
| `copy()` | 深拷贝 |
| `crossover(parent1, parent2)` | 静态方法，子树交换交叉 |
| `mutate(mutation_rate, fitness)` | 变异操作 |

**属性：**

| 属性 | 说明 |
|------|------|
| `root` | 根节点 |
| `fitness` | 适应度值 |
| `mdl_score` | MDL 复杂度评分 |

### 节点类型

#### `OperationNode`

操作节点，支持 `ADD, SUB, MUL, DIV, SIN, COS, EXP, LOG, NEG`。

```python
from neoglyph import OperationNode

node = OperationNode('ADD', left_child, right_child)
node = OperationNode('SIN', left_child)  # 一元操作
```

> 一元操作 (SIN/COS/EXP/LOG/NEG) 只需要 left 参数。

**方法：**

| 方法 | 说明 |
|------|------|
| `evaluate(x)` | 求值 |
| `to_expression()` | 转表达式 |
| `to_vm_code(var_map)` | 转 VM 代码 |
| `simplify()` | 符号简化，返回简化后的节点 |
| `copy()` | 深拷贝 |
| `get_depth()` | 获取深度 |
| `get_size()` | 获取子节点数 |
| `get_nodes()` | 获取所有节点 |

#### `VariableNode`

变量节点，支持单变量和多变量。

```python
from neoglyph import VariableNode

x = VariableNode('x')    # 单变量
y = VariableNode('y')    # 多变量
```

#### `ConstantNode`

常数节点。

```python
from neoglyph import ConstantNode

c = ConstantNode(2.0)
c = ConstantNode(3.14159)  # π
```

---

## 自定义算子

通过 `OperationNode.register_operator()` 注册自定义算子：

```python
from neoglyph import OperationNode

# 注册一元算子
OperationNode.register_operator('ABS', lambda a: np.abs(a), is_unary=True)
OperationNode.register_operator('SQUARE', lambda a: a**2, is_unary=True)
OperationNode.register_operator('SQRT', lambda a: np.sqrt(np.maximum(a, 0)), is_unary=True)

# 注册二元算子
OperationNode.register_operator('MAX', lambda a, b: np.maximum(a, b))
OperationNode.register_operator('MIN', lambda a, b: np.minimum(a, b))

# 使用
node = OperationNode('ABS', VariableNode('x'))
```

---

## 帕累托最优

### `ParetoFront`

帕累托最优筛选，同时优化误差和复杂度。

```python
from neoglyph import ParetoFront

# 获取帕累托前沿
front = ParetoFront.select(population, X, y)
for candidate in front[:3]:
    print(f"{candidate['expression']}: MSE={candidate['mse']:.6f}, "
          f"complexity={candidate['complexity']}")

# 按复杂度预算筛选
candidates = ParetoFront.best_by_complexity_budget(population, X, y, max_complexity=10)
```

---

## 可视化

### `NeoGlyphVM`

栈式虚拟机，支持 20+ 指令和自动微分。

```python
from neoglyph import NeoGlyphVM

vm = NeoGlyphVM(verbose=False)  # verbose=True 时 PRINT 指令输出
```

**支持的指令：**

| 类别 | 指令 |
|------|------|
| 算术 | ADD, SUB, MUL, DIV, POW, NEG |
| 数学函数 | SIN, COS, EXP, LOG, RELU |
| 栈操作 | PUSH, POP, DUP |
| 变量 | LOAD, STORE |
| 控制流 | JMP, JMP_IF, HALT |
| 自动微分 | TAPE, UNTAPE, GRAD |
| 矩阵 | MATMUL, SHAPE |
| 调试 | PRINT |

**方法：**

| 方法 | 说明 |
|------|------|
| `execute(program)` | 执行指令列表 |
| `reset()` | 重置 VM 状态 |
| `get_profile_report()` | 获取性能报告 |
| `get_fitness_metrics()` | 获取 fitness 相关指标 |

### `Tensor`

张量类，支持自动微分。

```python
from neoglyph import Tensor

t = Tensor([1.0, 2.0, 3.0])
t = Tensor(np.array([1.0, 2.0]))
```

---

## 可视化

### `plot_fit_curve(regressor, X, y, title, save_path)`

绘制拟合曲线：真实数据点 vs 预测曲线。

```python
from neoglyph import plot_fit_curve

reg.fit(X, y)
plot_fit_curve(reg, X, y, title="My Model", save_path="fit.png")
```

### `plot_evolution_history(history, title, save_path)`

绘制进化损失曲线：每代最佳/平均 fitness。

```python
from neoglyph import plot_evolution_history

reg.fit(X, y)
plot_evolution_history(reg.history_, title="Evolution Progress", save_path="loss.png")
```

### `plot_expression_tree(genome, title, save_path)`

绘制表达式树结构图。

```python
from neoglyph import plot_expression_tree

reg.fit(X, y)
plot_expression_tree(reg.best_genome_, title="Best Formula", save_path="tree.png")
```

### `plot_pareto_front(pareto_genomes, save_path)`

绘制帕累托前沿：误差 vs 复杂度。

```python
from neoglyph import ParetoFront, plot_pareto_front

front = ParetoFront.select(population, X, y)
plot_pareto_front(front, save_path="pareto.png")
```

### `print_tree(genome)`

在终端打印 ASCII 表达式树。

```python
from neoglyph import print_tree

print_tree(tree)
# └── EXPR
#     └── ADD
#         ├── MUL
#         │   ├── Var(x)
#         │   └── Const(2.00)
#         └── Const(1.00)
```

---

## 向后兼容

以下类保留用于向后兼容，新代码请使用 `EvolutionEngine(genome_type="tree")`：

```python
from neoglyph import Genome, GeneticOptimizer, ArchiveMemory
from neoglyph import TreeEvolutionEngine  # → EvolutionEngine(genome_type="tree")
from neoglyph import ParallelSymbolicRegressor  # → SymbolicRegressor(parallel=True)
```