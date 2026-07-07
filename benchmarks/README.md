# NeoGlyph Benchmark System

用于测试NeoGlyph在不同问题类型上的进化能力。

## Benchmark 类型

### 1. Linear（线性函数）
- `y = x + 3`：基础常数发现
- `y = 2x + 1`：系数+常数
- `y = 5x - 7`：复杂系数+负常数

### 2. Polynomial（多项式函数）
- `y = x*x`：基础平方
- `y = x*x + 2x + 1`：完全二次式

### 3. Control（控制流）
- `if x>0: y=1 else y=0`：条件判断

## 运行方法

```bash
python benchmarks/run_benchmarks.py
```

## 输出

生成 `benchmark_report.json`，包含：

- 每个benchmark的初始/最终fitness
- 进化代数
- 最佳Genome和解码脚本
- 预测误差
- 进化历史

## 目标

建立NeoGlyph能力测试标准，验证：
- 常数项发现能力
- 多项式表达能力
- 控制流进化能力