# NeoGlyph

<p align="center">
  <b>让 AI 自己发现数学公式</b>
  <br>
  <i>符号回归 · 程序进化 · 可解释 AI</i>
</p>

---

## 这是什么？

你有没有遇到过这种情况——手里有一堆数据，想找出背后的规律，但如果是线性回归，它只能给你一条直线；如果是神经网络，它给你一堆看不懂的权重。

**NeoGlyph 做的事不一样：它从数据中进化出人类可读的数学公式。**

给你看一个例子。给定 7 个数据点 `(-5,-9), (-3,-5), (-1,-1), (0,1), (1,3), (3,7), (5,11)`，NeoGlyph 通过进化算法，在 150 代内自动发现了隐藏的公式：

```
Gen   0: fitness=0.5392  expr=2 * x
Gen  30: fitness=1.0245  expr=2 * x + 1
Gen  60: fitness=1.0245  expr=2 * x + 1
Gen  90: fitness=1.0245  expr=2 * x + 1
Gen 120: fitness=1.0245  expr=2 * x + 1
Gen 149: fitness=1.0245  expr=2 * x + 1

✅ 最终发现: 2 * x + 1
MSE: 0.00000000
```

30 代就找到了精确答案，之后一直稳定保持。整个过程没有手动编程——全靠进化。

## 核心演示

直接看 5 个核心能力：

```
============================================================
  演示 1：手动构建表达式树
============================================================
  表达式: 2 * x + 1
  f(-3) = -5
  f( 0) = 1
  f( 5) = 11

============================================================
  演示 2：符号简化器
============================================================
  3*x + 2*x + 1        →  5 * x + 1
  x + x + x + 5        →  3 * x + 5
  sin(0) + 2*1         →  2

============================================================
  演示 3：自动发现公式 f(x) = 2x + 1
============================================================
  Gen   0: fitness=0.5392  expr=2 * x
  Gen  30: fitness=1.0245  expr=2 * x + 1
  Gen  60: fitness=1.0245  expr=2 * x + 1
  Gen  90: fitness=1.0245  expr=2 * x + 1
  Gen 120: fitness=1.0245  expr=2 * x + 1
  Gen 149: fitness=1.0245  expr=2 * x + 1

  最终发现: 2 * x + 1
  MSE: 0.00000000

============================================================
  演示 4：多变量符号回归 f(x,y) = x + y
============================================================
  发现表达式: y + x
  R² = 1.0000

============================================================
  演示 5：表达式 → VM 指令
============================================================
  表达式: 2 * x + 1
  VM 代码:
    LOAD a
    PUSH 2.0
    MUL
    PUSH 1.0
    ADD
============================================================
```

> 运行 `python examples/demo_readme.py` 即可复现以上所有输出。

## 为什么用 NeoGlyph？

| 场景 | 传统方法 | NeoGlyph |
|------|----------|----------|
| 从数据中找规律 | 猜函数形式 → 手动拟合 | 自动搜索，直接给出公式 |
| 解释黑箱模型 | 看输入输出，靠直觉 | 进化出等价的数学表达式 |
| 发现物理定律 | 假设公式 → 验证 | 数据驱动，自动发现 |
| 符号回归 | 固定模板 | 灵活树结构，支持任意嵌套 |

## 快速开始

```bash
pip install -e .
```

依赖仅需 `numpy`。

```python
from neoglyph import TreeGenome, ConstantNode, VariableNode, OperationNode

# 构建表达式树: 2*x + 1
x = VariableNode('x')
tree = TreeGenome(OperationNode('ADD',
    OperationNode('MUL', x, ConstantNode(2.0)),
    ConstantNode(1.0)
))

print(tree.to_expression())  # "2 * x + 1"
print(tree.root.evaluate(5))  # 11.0
```

自动发现公式：

```python
from neoglyph import SymbolicRegressor
import numpy as np

X = np.array([-5, -3, -1, 0, 1, 3, 5], dtype=np.float64)
y = 2 * X + 1

reg = SymbolicRegressor(pop_size=60, max_depth=2, generations=100, random_state=42)
reg.fit(X, y)
print(reg.expression())  # e.g. "2 * x + 1"
print(f"R² = {reg.score(X, y):.4f}")
```

多变量回归：

```python
X = np.array([[1,2], [3,4], [5,6]], dtype=np.float64)  # 两列 = 两个变量
y = 2 * X[:, 0] + 3 * X[:, 1]  # f(x,y) = 2x + 3y

reg = SymbolicRegressor(pop_size=60, max_depth=2, generations=100)
reg.fit(X, y)
print(reg.expression())  # e.g. "3 * y + 2 * x"
```

## 特性

- **Tree Genome** — 表达式树直接映射数学公式，支持单变量和多变量
- **符号简化器** — 自动合并同类项（`3x+2x → 5x`）、规范化常数、消除冗余
- **统一进化引擎** — 单一 `EvolutionEngine`，linear/tree 双模式，可配置策略
- **高级进化策略** — 适应度共享（Fitness Sharing）、岛模型（Island Model）、早停、精英存档
- **完整 VM** — 自定义栈式虚拟机，支持 20+ 指令和自动微分
- **应用层** — `SymbolicRegressor`、`PhysicsDiscoverer`，统一 `fit/predict/score` 接口

## 项目结构

```
NeoGlyph/
├── neoglyph/                  # 核心模块
│   ├── vm.py                 # 栈式虚拟机 + 自动微分
│   ├── tensor.py             # 张量
│   ├── ops.py                # VM 操作符
│   ├── genome.py             # TreeGenome + 符号简化
│   ├── genome_linear.py      # 线性 Genome（向后兼容）
│   ├── evolution.py          # 统一进化引擎 + 高级策略
│   ├── evolution_advanced.py # 高级进化特性
│   ├── applications.py       # 应用层
│   └── profiler.py           # 性能分析
├── tests/                     # 126 个测试
├── examples/                  # 12 个 Demo 脚本
├── benchmarks/                # 基准测试
└── docs/                      # 文档
```

## 运行测试

```bash
python -m unittest discover tests/ -v
```

126 个测试全部通过。

## 版本历史

- **v4.0** — 统一 Evolution API + Application API，FitnessSharing、IslandModel、多变量支持、evaluate_array 向量化
- **v3.3** — 同类项合并、简化缓存、智能随机生成
- **v3.2** — 符号简化器、可读性评分、常数规范化
- **v3.1** — Tree Genome、结构化变异、Archive Memory
- **v3.0** — 并行评估、Curriculum Learning、Discovery Score
- **v2.0** — 线性 Genome + 基础进化引擎
- **v1.0** — VM + Tensor + AutoGrad 基础架构

## ☕ 请作者喝杯咖啡

如果这个项目帮你省了时间、解决了问题，可以请我喝杯咖啡提神——这会让我很开心。

![支付宝打赏](docs/alipay_qr.png)

---

<p align="center">
  <i>这是我个人的业余作品，我会尽量关注 Issue，但可能无法及时回复每个问题。<br>
  如果遇到紧急问题，请按 Issue 模板详细描述，这样我能更快定位。</i>
</p>