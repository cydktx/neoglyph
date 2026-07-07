# NeoGlyph 文档

欢迎来到 NeoGlyph 文档。这里包含了从入门到进阶的所有资料。

## 快速导航

| 文档 | 内容 |
|------|------|
| [符号回归指南](guide_symbolic_regression.md) | 从数据中自动发现数学公式 |
| [物理公式发现](guide_physics_discovery.md) | 从实验数据发现物理定律 |
| [可解释 AI](guide_explainable_ai.md) | 将黑箱模型翻译成数学公式 |
| [API 参考](api_reference.md) | 所有类和方法的详细说明 |
| [配套工具链](toolchain.md) | 与 NeoGlyph 配合使用的工具生态 |

## 5 分钟快速上手

```bash
git clone https://github.com/cydktx/neoglyph.git
cd neoglyph
pip install -e .
```

```python
import numpy as np
from neoglyph import SymbolicRegressor

X = np.array([-5, -3, -1, 0, 1, 3, 5], dtype=np.float64)
y = 2 * X + 1

reg = SymbolicRegressor(pop_size=60, max_depth=2, generations=100, random_state=42)
reg.fit(X, y)
print(reg.expression())  # "2 * x + 1"
```

## 核心概念

- **表达式树 (Tree Genome)** — 数学表达式表示为树结构，每个节点是操作、变量或常数
- **符号简化器** — 自动合并同类项、规范化表达式
- **进化引擎** — 通过选择、交叉、变异进化出更优的表达式
- **VM 执行** — 自定义栈式虚拟机执行表达式，支持自动微分

## 架构

```
用户 API 层
  ├── SymbolicRegressor  (符号回归)
  ├── PhysicsDiscoverer  (物理发现)
  └── BaseApplication    (自定义应用)

进化引擎层
  ├── EvolutionEngine    (统一引擎)
  ├── FitnessSharing     (适应度共享)
  ├── IslandModel        (岛模型)
  └── EarlyStopper       (早停)

基因组层
  ├── TreeGenome         (表达式树)
  ├── OperationNode      (操作节点)
  ├── VariableNode       (变量节点)
  └── ConstantNode       (常数节点)

执行层
  ├── NeoGlyphVM         (栈式虚拟机)
  ├── Tensor             (张量)
  └── ops                (操作符)
```