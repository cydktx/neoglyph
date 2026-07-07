# 配套工具链

NeoGlyph 不是孤立的工具。以下是与它配合使用的推荐工具生态。

## 数据准备

| 工具 | 用途 | 安装 |
|------|------|------|
| **numpy** | 数组运算，NeoGlyph 唯一依赖 | `pip install numpy` |
| **pandas** | 读取 CSV/Excel 数据，数据清洗 | `pip install pandas` |
| **scipy** | 科学计算，优化算法 | `pip install scipy` |

```python
import pandas as pd
import numpy as np
from neoglyph import SymbolicRegressor

# 从 CSV 读取数据
df = pd.read_csv("experiment_data.csv")
X = df['input'].values.astype(np.float64)
y = df['output'].values.astype(np.float64)

reg = SymbolicRegressor(pop_size=60, max_depth=2, generations=100)
reg.fit(X, y)
print(reg.expression())
```

## 可视化

| 工具 | 用途 | 安装 |
|------|------|------|
| **matplotlib** | 基础绘图：进化曲线、拟合对比 | `pip install matplotlib` |
| **seaborn** | 更美观的统计图表 | `pip install seaborn` |
| **graphviz** | 表达式树可视化 | `pip install graphviz` |

### 进化曲线

```python
import matplotlib.pyplot as plt

reg.fit(X, y)

# 提取历史
gens = [h['generation'] for h in reg.history_]
best = [h['best_fitness'] for h in reg.history_]
avg = [h['avg_fitness'] for h in reg.history_]

plt.figure(figsize=(10, 5))
plt.plot(gens, best, label='Best Fitness', linewidth=2)
plt.plot(gens, avg, label='Avg Fitness', alpha=0.7)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Evolution Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('evolution_curve.png', dpi=150)
```

### 拟合对比

```python
# 对比真实值和预测值
y_pred = reg.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title(f'R² = {reg.score(X, y):.4f}')
plt.tight_layout()
plt.savefig('fit_comparison.png', dpi=150)
```

### 表达式树可视化

```python
from neoglyph import TreeGenome, ConstantNode, VariableNode, OperationNode

# 构建表达式树
x = VariableNode('x')
tree = TreeGenome(OperationNode('ADD',
    OperationNode('MUL', x, ConstantNode(2.0)),
    ConstantNode(1.0)
))

# 打印树结构
def print_tree(node, indent=0):
    prefix = "  " * indent + ("├─ " if indent > 0 else "")
    if node.node_type == 'constant':
        print(f"{prefix}Const({node.value})")
    elif node.node_type == 'variable':
        print(f"{prefix}Var({node.name})")
    else:
        print(f"{prefix}{node.op}")
        if node.left:
            print_tree(node.left, indent + 1)
        if node.right:
            print_tree(node.right, indent + 1)

print_tree(tree.root)
# 输出:
# ADD
#   ├─ MUL
#   │   ├─ Var(x)
#   │   ├─ Const(2.0)
#   ├─ Const(1.0)
```

## 机器学习对比

| 工具 | 用途 | 场景 |
|------|------|------|
| **scikit-learn** | 传统 ML 基线 | 与 NeoGlyph 的符号回归结果对比 |
| **PyTorch / TensorFlow** | 神经网络 | 可解释 AI 流程中的黑箱模型 |
| **XGBoost / LightGBM** | 梯度提升 | 另一类黑箱，也可用 NeoGlyph 解释 |

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from neoglyph import SymbolicRegressor

# 对比：线性回归 vs NeoGlyph
X = np.linspace(-5, 5, 50)
y = X ** 2 + 2 * X + 1  # 二次函数

# 线性回归（注定失败）
lr = LinearRegression()
lr.fit(X.reshape(-1, 1), y)
lr_pred = lr.predict(X.reshape(-1, 1))

# NeoGlyph（自动发现二次关系）
reg = SymbolicRegressor(pop_size=60, max_depth=3, generations=100, random_state=42)
reg.fit(X, y)
sym_pred = reg.predict(X)

print(f"线性回归 R²: {r2_score(y, lr_pred):.4f}")
print(f"NeoGlyph  R²: {r2_score(y, sym_pred):.4f}")
print(f"NeoGlyph 公式: {reg.expression()}")
```

## 部署与集成

| 工具 | 用途 |
|------|------|
| **joblib / pickle** | 模型序列化 |
| **Flask / FastAPI** | 将发现的公式部署为 API |
| **Jupyter Notebook** | 交互式探索和演示 |

```python
# 部署为 FastAPI 服务
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from neoglyph import SymbolicRegressor

app = FastAPI()
reg = SymbolicRegressor()
reg.load("discovered_formula.pkl")

class PredictRequest(BaseModel):
    x: list[float]

@app.post("/predict")
def predict(req: PredictRequest):
    X = np.array(req.x, dtype=np.float64)
    return {"predictions": reg.predict(X).tolist()}
```

## 完整工作流

```
原始数据 (CSV/Excel)
    │
    ├── pandas 读取、清洗
    │
    ├── matplotlib 可视化探索
    │
    ├── NeoGlyph 符号回归
    │   ├── SymbolicRegressor.fit()
    │   ├── reg.expression() → 人类可读公式
    │   └── reg.save() → 保存模型
    │
    ├── scikit-learn 对比验证
    │   └── r2_score, MSE, MAE
    │
    └── Flask/FastAPI 部署
        └── 公式变成 API 服务
```