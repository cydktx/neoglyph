# NeoGlyph

<p align="center">
  <b>给 AI 一堆数据，还你一个公式。</b>
</p>

---

<p align="center">
  <i>神经网络给你权重，线性回归给你直线——NeoGlyph 给你 <b>人类可读的数学公式</b>。<br>
  符号回归 · 程序进化 · 可解释 AI</i>
</p>

---

## 亲眼看看

打开终端，运行 `python examples/demo_hero.py`，你会看到这个：

```
╔══════════════════════════════════════════════════════╗
║           NeoGlyph — 让 AI 发现数学公式              ║
║           符号回归 · 程序进化 · 可解释 AI             ║
╚══════════════════════════════════════════════════════╝

📊 输入数据:  (x, y) = (-5,-9), (-3,-5), (-1,-1), (0,1), (1,3), (3,7), (5,11)

  Gen   0  [██████████░░░░░░░░░░]  fitness=0.5392  2 * x
  Gen  10  [████████████████████]  fitness=1.0245  2 * x + 1  ← 🎯 发现目标！
  Gen  20  [████████████████████]  fitness=1.0245  2 * x + 1  ← 🎯 发现目标！
  Gen  30  [████████████████████]  fitness=1.0245  2 * x + 1  ← 🎯 发现目标！
  Gen  50  [████████████████████]  fitness=1.0245  2 * x + 1  ← 🎯 发现目标！
  Gen  80  [████████████████████]  fitness=1.0245  2 * x + 1  ← 🎯 发现目标！
  Gen 149  [████████████████████]  fitness=1.0245  2 * x + 1  ← 🎯 发现目标！

  ✅ 最终发现: 2 * x + 1
  📐 均方误差: 0.00000000
  ⏱️  耗时: 150 代 (约 2 秒)

  💡 整个过程没有手动编程 — 全靠进化算法自动搜索。
```

**第 10 代就找到了 `2x+1`，之后一直稳定。** 你只给了 7 个数据点，没有写任何规则，没有指定函数形式。AI 自己"进化"出了答案。

---

## 5 分钟上手指南

> 从零开始，只要 5 分钟。

### 第 1 步：安装（30 秒）

```bash
git clone https://github.com/cydktx/neoglyph.git
cd neoglyph
pip install -e .
```

唯一依赖：`numpy`（Python 自带以外的唯一第三方库）。

### 第 2 步：跑起来（1 分钟）

新建一个 Python 文件，粘贴以下代码，直接运行：

```python
import numpy as np
from neoglyph import SymbolicRegressor

# 1. 准备数据：7 个点，藏着 y = 2x + 1 的规律
X = np.array([-5, -3, -1, 0, 1, 3, 5], dtype=np.float64)
y = 2 * X + 1

# 2. 创建进化引擎，让它自己找公式
reg = SymbolicRegressor(
    pop_size=60,        # 60 个候选公式同时进化
    max_depth=2,        # 公式复杂度上限
    generations=100,    # 进化 100 代
    random_state=42,
)
reg.fit(X, y)

# 3. 看结果
print(f"发现的公式: {reg.expression()}")
print(f"R² = {reg.score(X, y):.4f}")
print(f"MSE = {np.mean((reg.predict(X) - y) ** 2):.8f}")
```

输出会是：

```
发现的公式: 2 * x + 1
R² = 1.0000
MSE = 0.00000000
```

### 第 3 步：换你自己的数据（3 分钟）

把 `X` 和 `y` 换成你自己的数据，NeoGlyph 会自动帮你发现公式。

```python
# 试试这些：
# 二次函数: y = x**2
# 多变量:   y = 2*x1 + 3*x2
# 物理公式: y = 0.5 * 9.8 * x**2  (自由落体)
```

多变量也支持：

```python
X = np.array([[1,2], [3,4], [5,6]], dtype=np.float64)
y = 2 * X[:, 0] + 3 * X[:, 1]  # f(x,y) = 2x + 3y

reg = SymbolicRegressor(pop_size=60, max_depth=2, generations=100)
reg.fit(X, y)
print(reg.expression())  # → "3 * y + 2 * x"
```

### 第 4 步：手动构建表达式树

你也可以不靠进化，直接手动构建表达式：

```python
from neoglyph import TreeGenome, ConstantNode, VariableNode, OperationNode

x = VariableNode('x')
tree = TreeGenome(OperationNode('ADD',
    OperationNode('MUL', x, ConstantNode(2.0)),
    ConstantNode(1.0)
))

print(tree.to_expression())   # "2 * x + 1"
print(tree.root.evaluate(5))  # 11.0

# 符号简化
print(tree.simplify().to_expression())  # "2 * x + 1"
```

---

## 解决了什么问题？

| 场景 | 传统方法 | NeoGlyph |
|------|----------|----------|
| 从数据中找规律 | 猜函数形式 → 手动拟合 → 反复试错 | 自动搜索，直接给出公式 |
| 解释黑箱模型 | 看输入输出，靠直觉猜 | 进化出等价的数学表达式 |
| 发现物理定律 | 假设公式 → 验证 → 改假设 | 数据驱动，自动发现 |
| 符号回归 | 固定模板，不支持嵌套 | 灵活树结构，支持任意嵌套和一元函数 |

---

## 为什么选 NeoGlyph？

与同类工具的对比：

| 特性 | NeoGlyph | gplearn | AI-Feynman |
|------|----------|---------|------------|
| 核心算法 | 树GP + 符号简化 | 树GP | 神经网络 + 符号回归 |
| 多变量支持 | 原生支持 | 需手动编码 | 支持 |
| 符号简化 | 同类项合并、常数规范化 | 无 | 无 |
| 自定义算子 | 一行注册 | 需修改源码 | 不支持 |
| 帕累托最优 | 内置 | 无 | 无 |
| 可解释 AI | 完整流程 | 无 | 核心功能 |
| 可视化 | 拟合曲线/进化损失/树 | 基础 | 无 |
| 依赖 | 仅 numpy | numpy + sklearn | PyTorch + 多个 |
| GPU 需求 | 不需要 | 不需要 | 需要 |
| 离线可用 | 完全离线 | 完全离线 | 依赖网络权重 |
| 最小样本量 | 5-10 个点 | 20+ 个点 | 100+ 个点 |

**NeoGlyph 适用于：**
- 小样本场景（几十个数据点就够了）
- 无 GPU 环境（笔记本、边缘设备）
- 完全离线（实验室、涉密环境）
- 需要可解释公式（科研论文、教育）
- 快速原型验证（从数据到公式，2 秒）

---

## 核心特性

- **Tree Genome** — 表达式树直接映射数学公式，支持单变量和多变量
- **符号简化器** — 自动合并同类项（`3x+2x → 5x`）、规范化常数、消除冗余
- **统一进化引擎** — 单一 `EvolutionEngine`，支持 linear/tree 双模式，可配置策略
- **高级进化策略** — 适应度共享（Fitness Sharing）、岛模型（Island Model）、早停、帕累托最优
- **自定义算子** — 一行代码注册新算子，如 `ABS`、`SQRT`、`MAX`
- **完整 VM** — 自定义栈式虚拟机，20+ 指令 + 自动微分
- **应用层** — `SymbolicRegressor`、`PhysicsDiscoverer`，统一 `fit/predict/score` 接口
- **可视化** — 拟合曲线、进化损失曲线、表达式树、帕累托前沿

## 文档

📚 完整文档请查看 [docs/](docs/index.md)：

| 文档 | 内容 |
|------|------|
| [符号回归指南](docs/guide_symbolic_regression.md) | 从数据中自动发现数学公式 |
| [物理公式发现](docs/guide_physics_discovery.md) | 从实验数据发现物理定律 |
| [可解释 AI](docs/guide_explainable_ai.md) | 将黑箱模型翻译成数学公式 |
| [API 参考](docs/api_reference.md) | 所有类和方法的详细说明 |
| [配套工具链](docs/toolchain.md) | 与 NeoGlyph 配合使用的工具生态 |

## 项目结构

```
NeoGlyph/
├── neoglyph/                  # 核心模块
│   ├── vm.py                 # 栈式虚拟机 + 自动微分
│   ├── tensor.py             # 张量
│   ├── ops.py                # VM 操作符
│   ├── genome.py             # TreeGenome + 符号简化 + 自定义算子
│   ├── genome_linear.py      # 线性 Genome（向后兼容）
│   ├── evolution.py          # 统一进化引擎 + 高级策略 + 帕累托
│   ├── evolution_advanced.py # 高级进化特性
│   ├── applications.py       # 应用层
│   ├── visualization.py      # 可视化（拟合曲线/进化损失/表达式树）
│   └── profiler.py           # 性能分析
├── tests/                     # 126 个测试
├── examples/                  # 13 个 Demo 脚本
├── benchmarks/                # 基准测试
└── docs/                      # 文档
```

## 运行测试

```bash
python -m unittest discover tests/ -v
```

126 个测试全部通过。

## 版本历史

- **v4.1** — 可视化模块、自定义算子接口、帕累托最优筛选、pip 打包完善
- **v4.0** — 统一 Evolution API + Application API，FitnessSharing、IslandModel、多变量支持、evaluate_array 向量化
- **v3.3** — 同类项合并、简化缓存、智能随机生成
- **v3.2** — 符号简化器、可读性评分、常数规范化
- **v3.1** — Tree Genome、结构化变异、Archive Memory
- **v3.0** — 并行评估、Curriculum Learning、Discovery Score
- **v2.0** — 线性 Genome + 基础进化引擎
- **v1.0** — VM + Tensor + AutoGrad 基础架构

---

## 反馈与贡献

遇到问题？有想法？欢迎提 Issue。

👉 **[提交 Issue](https://github.com/cydktx/neoglyph/issues/new?template=ISSUE_TEMPLATE.md)**

请尽量附上：
- 你使用的 Python 和 numpy 版本
- 能复现问题的最小代码
- 期望的结果 vs 实际的结果

如果你有代码贡献，请参考 **[PR 模板](https://github.com/cydktx/neoglyph/blob/main/.github/PULL_REQUEST_TEMPLATE.md)**。

---

## ☕ 为船长加杯咖啡

<p align="center">
  <i>我是这个项目的"船长"，AI 是我的水手。<br>
  如果这艘船帮到了您，为我加杯咖啡燃料，我会很开心！</i>
</p>

<p align="center">
  <img src="docs/alipay_qr.png" alt="支付宝打赏" width="220">
</p>

---

<p align="center">
  <sub>这是我个人的业余作品，我会尽量关注 Issue，但可能无法及时回复每个问题。<br>
  如果遇到紧急问题，请按 Issue 模板详细描述，这样我能更快定位。</sub>
</p>