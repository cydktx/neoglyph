# 可解释 AI 指南

将黑箱模型（如神经网络）学到的知识，翻译成人类可读的数学公式。

## 为什么需要可解释 AI？

神经网络可以拟合任何函数，但你看不懂它学了什么：

```
Model: Sequential(
  Dense(64, relu),
  Dense(64, relu),
  Dense(1)
)
参数: 4,353 个浮点数
```

NeoGlyph 的做法：用神经网络拟合数据，再用符号回归提取公式。

```
黑箱: 4,353 个权重 → 白箱: y = 2x + 1
```

## 完整流程

### 第一步：训练神经网络（黑箱）

```python
import numpy as np

# 数据：隐藏函数 f(x) = x² + 2x + 1
X = np.linspace(-5, 5, 200).reshape(-1, 1)
y = X.ravel() ** 2 + 2 * X.ravel() + 1

# 划分训练/测试
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
```

### 第二步：用符号回归提取公式（白箱）

```python
from neoglyph import SymbolicRegressor

# 用神经网络的预测作为目标
nn_pred = trained_model.predict(X_train).ravel()

reg = SymbolicRegressor(
    pop_size=100, max_depth=3, generations=200,
    random_state=42,
)
reg.fit(X_train.ravel(), nn_pred)

# 对比：神经网络预测 vs 发现的公式
print(f"发现的公式: {reg.expression()}")

# 在测试集上验证
nn_test_pred = trained_model.predict(X_test).ravel()
sym_test_pred = reg.predict(X_test.ravel())

nn_mse = np.mean((nn_test_pred - y_test) ** 2)
sym_mse = np.mean((sym_test_pred - y_test) ** 2)

print(f"神经网络 MSE: {nn_mse:.6f}")
print(f"符号公式 MSE: {sym_mse:.6f}")
```

## 渐进式示例

完整的可解释 AI demo 在 `examples/demo_explainable_ai.py`，包含 4 个难度递增的示例：

| 示例 | 目标函数 | 难度 | 说明 |
|------|---------|------|------|
| 1 | `x² + 2x + 1` | 简单 | 验证基础能力 |
| 2 | `x + sin(x)` | 中等 | 含三角函数 |
| 3 | `x² + sin(3x) + 2` | 困难 | 当前极限 |
| 4 | `x + y` | 多变量 | 新能力 |

运行：

```bash
python examples/demo_explainable_ai.py
```

## 关键技巧

### 1. 训练/评估数据分离

神经网络在训练集上训练，符号回归在训练集上搜索，最后在测试集上对比：

```python
X_train, X_test = X[:160], X[160:]
y_train, y_test = y[:160], y[160:]

# NN 训练
model.fit(X_train, y_train, epochs=500, verbose=0)

# 符号回归 — 用 NN 的预测作为目标
nn_pred = model.predict(X_train).ravel()
reg.fit(X_train.ravel(), nn_pred)

# 在测试集上评估
print(f"R² (test) = {reg.score(X_test.ravel(), y_test):.4f}")
```

### 2. 神经网络要足够大

太小的网络学不到真实函数，太大的网络过拟合。推荐：

- 隐藏层：2-3 层
- 每层神经元：32-64
- 激活函数：ReLU 或 tanh
- 学习率衰减：有帮助

### 3. 符号回归参数

- 神经网络拟合得好 → 符号回归可以更激进（更大 pop_size）
- 神经网络拟合得差 → 符号回归需要更保守（更小 max_depth）

## 限制

- 神经网络本身的误差会传递给符号回归
- 非常复杂的函数（如分形）可能无法用简单表达式表示
- 当前的符号回归不支持分段函数（如 if-then-else）