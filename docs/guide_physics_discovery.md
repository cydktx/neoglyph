# 物理公式发现指南

从实验数据中自动发现物理定律，是 NeoGlyph 最令人兴奋的应用。

## 核心理念

传统的物理发现流程：

```
观察现象 → 提出假设 → 设计实验 → 收集数据 → 尝试拟合 → 修正假设 → ...
```

NeoGlyph 的流程：

```
收集数据 → 喂给 NeoGlyph → 得到公式
```

## 基础用法

```python
import numpy as np
from neoglyph import PhysicsDiscoverer

# 实验数据：自由落体 h = 0.5 * g * t²
t = np.linspace(0, 3, 20)           # 时间 (秒)
h = 0.5 * 9.8 * t ** 2              # 高度 (米)

# 发现公式
discoverer = PhysicsDiscoverer(
    pop_size=150,       # 更大种群，因为公式更复杂
    max_depth=5,        # 更深，支持嵌套
    generations=300,    # 更多代数
    random_state=42,
)

result = discoverer.discover(t, h)

print(f"发现公式: {result['expression']}")
print(f"R² = {result['r2_score']:.4f}")
print(f"MSE = {result['mse']:.6f}")
print(f"MAE = {result['mae']:.6f}")
print(f"复杂度: {result['complexity']}")
```

## 更多示例

### 胡克定律：F = kx

```python
x = np.linspace(0.1, 1.0, 15)    # 伸长量 (m)
F = 50.0 * x                       # 弹力 (N)

discoverer = PhysicsDiscoverer(
    pop_size=120, max_depth=3, generations=200,
    random_state=42,
)
result = discoverer.discover(x, F)
print(result['expression'])  # 应接近 "50 * x"
```

### 动能公式：E = 0.5mv²

```python
v = np.linspace(1, 10, 20)        # 速度 (m/s)
m = 2.0                            # 质量 (kg)
E = 0.5 * m * v ** 2              # 动能 (J)

# 多变量：v 和 m 都是变量
X = np.column_stack([v, np.full_like(v, m)])
y = E

discoverer = PhysicsDiscoverer(
    pop_size=200, max_depth=5, generations=400,
    random_state=42,
)
result = discoverer.discover(X, y)
print(result['expression'])  # 应接近 "0.5 * m * v^2" 或等价形式
```

### 单摆周期：T = 2π√(L/g)

```python
L = np.linspace(0.5, 2.0, 20)     # 摆长 (m)
g = 9.8                            # 重力加速度
T = 2 * np.pi * np.sqrt(L / g)    # 周期 (s)

discoverer = PhysicsDiscoverer(
    pop_size=200, max_depth=5, generations=500,
    random_state=42,
)
result = discoverer.discover(L, T)
print(result['expression'])
```

## 参数调优建议

| 场景 | pop_size | max_depth | generations |
|------|----------|-----------|-------------|
| 简单线性关系 (F=kx) | 80-120 | 2-3 | 100-200 |
| 二次关系 (E=½mv²) | 120-200 | 3-4 | 200-400 |
| 含根号/三角函数 | 200-400 | 5-6 | 400-800 |
| 多变量复杂公式 | 200-500 | 5-7 | 500-1000 |

## 结果解读

`discover()` 返回的字典包含：

| 字段 | 含义 |
|------|------|
| `expression` | 发现的数学表达式 |
| `r2_score` | R² 决定系数 (1.0 = 完美) |
| `mse` | 均方误差 |
| `mae` | 平均绝对误差 |
| `complexity` | 表达式复杂度 (节点数) |
| `history` | 进化历史 (每代 fitness) |
| `genome` | 最佳 TreeGenome 对象 |

## 手动构建 + 验证

```python
from neoglyph import TreeGenome, ConstantNode, VariableNode, OperationNode

# 构建 F = kx
k = ConstantNode(50.0)
x = VariableNode('x')
formula = TreeGenome(OperationNode('MUL', k, x))

# 验证
for val in [0.1, 0.5, 1.0]:
    print(f"F({val}) = {formula.root.evaluate(val):.1f}")

# 符号简化
print(formula.simplify().to_expression())
```

## 注意事项

- 数据质量 >> 算法参数。噪声大的数据需要更多代数和更大种群
- 物理公式通常有量纲，NeoGlyph 只处理数值关系
- 发现的公式是"拟合"结果，不一定是真正的物理定律
- 建议用训练集和测试集分别验证泛化能力