# NeoGlyph

**符号回归与程序进化系统** — 让AI自己发现数学公式。

NeoGlyph 通过遗传算法自动进化程序，从数据中发现简洁可读的数学表达式。

## ✨ 特性

- **Tree Genome** — 结构化树状基因组，直接映射数学表达式，支持多变量
- **符号简化器** — 自动合并同类项、规范化表达式，输出标准形式
- **统一进化引擎** — 单一 EvolutionEngine，支持 linear/tree 双模式，并行评估
- **高级进化策略** — 适应度共享、岛模型、早停机制、精英存档
- **完整VM** — 自定义栈式虚拟机，支持20+指令和自动微分
- **应用层** — SymbolicRegressor、PhysicsDiscoverer，统一 fit/predict/score 接口

## 🚀 快速开始

### 安装

```bash
pip install -e .
```

依赖：`numpy`

### 5分钟上手

```python
from neoglyph import TreeGenome, ConstantNode, VariableNode, OperationNode

# 手动构建表达式：y = 2*x + 1
x = VariableNode('x')
mul = OperationNode('MUL', x, ConstantNode(2.0))
add = OperationNode('ADD', mul, ConstantNode(1.0))
tree = TreeGenome(add)

print(tree.to_expression())  # "2 * x + 1"

# 求值
print(tree.root.evaluate(5))  # 11.0
```

### 自动进化发现公式

```bash
python examples/evolution_core_3_demo.py
```

示例输出：
```
🎯 Target: y = 2*x + 1

Generation 0:  Best: 2 * x       (fitness=0.57)
Generation 20: Best: 2 * x + 1   (fitness=1.04)

✅ Best Expression: 2 * x + 1
🎉 SUCCESS! Average error: 0.0000
```

## 📁 项目结构

```
NeoGlyph/
├── neoglyph/                  # 核心模块
│   ├── __init__.py           # 入口
│   ├── vm.py                 # 栈式虚拟机 + 自动微分
│   ├── tensor.py             # 张量基础
│   ├── ops.py                # VM操作符
│   ├── genome.py             # TreeGenome + 符号简化
│   ├── genome_linear.py      # 线性Genome（向后兼容）
│   ├── evolution.py          # 统一进化引擎 + 高级策略
│   ├── evolution_advanced.py # 高级进化特性
│   ├── applications.py       # 应用层（SymbolicRegressor等）
│   └── profiler.py           # 性能分析
│
├── tests/                     # 单元测试（126个）
├── examples/                  # Demo脚本
│   ├── evolution_core_3_demo.py
│   ├── demo_explainable_ai.py
│   ├── demo_physics.py
│   └── ...
├── benchmarks/                # 基准测试
└── docs/                      # 文档
```

## 🧬 核心概念

### Tree Genome（树基因组）

程序表示为表达式树，每个节点是操作、变量或常数：

```
    ADD
   /   \
  MUL   1
 /   \
x     2
```

对应表达式：`2 * x + 1`

### 符号简化

自动将复杂表达式规范化为标准形式：

```
x + x + x + 1 + 2  →  3 * x + 3
3 * x + 2 * x + 1  →  5 * x + 1
x - (-2)           →  x + 2
```

### 进化算法

- **精英保留** — 保留Top 20%优秀个体
- **锦标赛选择** — 选择父代
- **结构变异** — 替换节点/添加分支/删除分支/优化常数
- **保护机制** — 高fitness节点降低变异概率

## 🔬 Benchmark

```
Linear:      y=x+3, y=2x+1, y=5x-7
Polynomial:  y=x*x, y=x*x+2x+1
Control:     if x>0 → 1 else 0
```

运行基准测试：
```bash
python benchmarks/run_benchmarks.py
```

## 🧪 运行测试

```bash
python -m unittest discover tests/ -v
```

126 个测试全部通过。

## 📊 版本历史

- **v4.0** — 统一 Evolution API + Application API，FitnessSharing、IslandModel、多变量支持、evaluate_array 向量化
- **v3.3** — Evolution Core 3.2：同类项合并、简化缓存、智能随机生成
- **v3.2** — Evolution Core 3.1：符号简化器、可读性评分、常数规范化
- **v3.1** — Evolution Core 3.0：Tree Genome、结构化变异、Archive Memory
- **v3.0** — Advanced Evolution：并行评估、Curriculum Learning、Discovery Score
- **v2.0** — 线性Genome + 基础进化引擎
- **v1.0** — VM + Tensor + AutoGrad 基础架构

## ☕ 支持作者

如果这个项目对你有帮助，欢迎请作者喝杯咖啡。

![支付宝打赏](docs/alipay_qr.png)

## 📝 License

MIT
