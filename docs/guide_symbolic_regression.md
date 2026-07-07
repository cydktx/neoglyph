# 符号回归指南

符号回归是 NeoGlyph 最核心的能力：从输入输出数据中自动发现数学表达式。

## 什么是符号回归？

传统回归（如线性回归）要求你预先指定函数形式（如 `y = ax + b`），然后拟合参数。符号回归不需要——它**自动搜索函数形式和参数**。

```
数据: (1,3), (2,5), (3,7), (4,9)
线性回归: 你告诉它 y = ax + b，它帮你找 a=2, b=1
符号回归: 它自己发现 y = 2x + 1
```

## 基础用法

```python
import numpy as np
from neoglyph import SymbolicRegressor

# 准备数据
X = np.array([1, 2, 3, 4, 5], dtype=np.float64)
y = 2 * X + 1  # 目标函数

# 创建回归器
reg = SymbolicRegressor(
    pop_size=60,        # 种群大小：每代 60 个候选公式
    max_depth=2,        # 最大深度：控制公式复杂度
    generations=100,    # 进化代数
    random_state=42,    # 可复现
)

# 拟合
reg.fit(X, y)

# 查看结果
print(reg.expression())   # 2 * x + 1
print(reg.score(X, y))    # R² = 1.0
print(reg.predict([6]))   # 13.0
```

## 参数调优

### 种群大小 (`pop_size`)

- **太小** (20-30)：收敛慢，容易陷入局部最优
- **适中** (50-100)：大多数场景的推荐值
- **太大** (200+)：计算量大，但可能更快找到解

### 最大深度 (`max_depth`)

- `1`：只能表示 `ax + b` 形式
- `2`：可以表示 `ax² + bx + c` 等二次函数
- `3`：支持更复杂的嵌套，如 `sin(ax) + bx`
- `4+`：任意复杂表达式，但搜索空间急剧增大

### 进化代数 (`generations`)

- 简单线性函数：50-100 代通常足够
- 二次函数：100-200 代
- 复杂非线性：200-500 代

### 变异率 (`mutation_rate`)

- 默认 0.3 适用于大多数场景
- 提高 (0.5)：探索更多可能性，但可能不稳定
- 降低 (0.1)：保守搜索，适合简单问题

## 多变量回归

```python
# f(x, y) = 2x + 3y
X = np.array([[1,2], [3,4], [5,6]], dtype=np.float64)
y = 2 * X[:, 0] + 3 * X[:, 1]

reg = SymbolicRegressor(pop_size=60, max_depth=2, generations=100)
reg.fit(X, y)
print(reg.expression())  # "3 * y + 2 * x"
```

NeoGlyph 自动检测变量数量，使用 `x, y, z, ...` 作为变量名。

## 进化历史

```python
reg.fit(X, y)

# 查看每代的最佳 fitness
for entry in reg.history_:
    print(f"Gen {entry['generation']}: "
          f"best={entry['best_fitness']:.4f}, "
          f"avg={entry['avg_fitness']:.4f}")
```

## 保存和加载

```python
# 保存
reg.save("my_model.pkl")

# 加载
from neoglyph import SymbolicRegressor
reg2 = SymbolicRegressor()
reg2.load("my_model.pkl")
print(reg2.expression())
```

## 高级用法：直接使用进化引擎

```python
from neoglyph import EvolutionEngine, TreeGenome
import numpy as np

X = np.array([-5, -3, -1, 0, 1, 3, 5], dtype=np.float64)
target_fn = lambda x: x ** 2 + 2 * x + 1

engine = EvolutionEngine(
    genome_type="tree",
    pop_size=100,
    max_depth=3,
    mutation_rate=0.3,
    random_state=42,
)

import random
for gen in range(200):
    for g in engine.population:
        g.calculate_fitness(X, target_fn)
    
    engine.population.sort(key=lambda g: g.fitness, reverse=True)
    best = engine.population[0]
    
    if gen % 40 == 0:
        print(f"Gen {gen:3d}: {best.to_expression()} (fitness={best.fitness:.4f})")
    
    # 精英 + 交叉 + 变异
    elite = [g.copy() for g in engine.population[:10]]
    new_pop = elite[:]
    while len(new_pop) < 100:
        p1 = random.choice(engine.population[:50])
        p2 = random.choice(engine.population[:50])
        child = TreeGenome.crossover(p1, p2)
        child.mutate(0.3, fitness=p1.fitness)
        new_pop.append(child)
    engine.population = new_pop
```

## 常见问题

### 为什么每次都得到不同的结果？

进化算法是随机的。设置 `random_state` 可以复现结果。

### 为什么发现的公式很奇怪？

- 增大 `pop_size` 和 `generations`
- 降低 `max_depth` 限制复杂度
- 增加数据点数量和质量

### 支持哪些函数？

支持 `+, -, *, /, sin, cos, exp, log, neg`。表达式树支持任意嵌套。