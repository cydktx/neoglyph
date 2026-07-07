"""
NeoGlyph 可解释 AI Demo v2
==========================

演示核心概念：把黑箱模型（神经网络）学到的知识，翻译成人类可读的数学公式。

渐进式示例：
  Example 1: f(x) = x² + 2x + 1      → 简单二次函数（应完整发现）
  Example 2: f(x) = x + sin(x)        → 中等难度（一元函数发现）
  Example 3: f(x) = x² + sin(3x) + 2  → 高难度（当前极限）
  Example 4: f(x,y) = x² + y²         → 多变量（新能力）

每个示例展示完整的 "黑箱 → 白箱" 转换流程。
"""

import numpy as np
import time
from neoglyph import SymbolicRegressor


# ============================================================================
# 微型神经网络（纯 numpy 实现）
# ============================================================================

class TinyNN:
    """2 层 MLP，tanh 激活，学习率衰减
    
    参数数量少、训练快，但行为接近黑箱——没人能直接从权重读出公式。
    """

    def __init__(self, input_dim=1, hidden_size=16):
        self.W1 = np.random.randn(input_dim, hidden_size) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(1)
        self.input_dim = input_dim

    def forward(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2.ravel()

    def backward(self, x, y, lr):
        m = len(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        pred = self.forward(x)
        dz2 = (pred - y).reshape(-1, 1) / m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - self.a1 ** 2)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, x, y, epochs=8000, lr=0.02, decay=0.9995, verbose=False):
        losses = []
        current_lr = lr
        for epoch in range(epochs):
            self.backward(x, y, current_lr)
            current_lr *= decay
            if epoch % 2000 == 0:
                pred = self.forward(x)
                loss = np.mean((pred - y) ** 2)
                losses.append(loss)
                if verbose:
                    print(f"  Epoch {epoch:5d}: loss={loss:.6f}, lr={current_lr:.6f}")
        return losses

    def param_count(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


# ============================================================================
# 符号回归：从黑箱提取公式
# ============================================================================

def explain_blackbox(nn_model, x_range, n_samples=100, pop_size=80, max_depth=4,
                     generations=200, verbose=False, random_state=42):
    """从神经网络采样数据中反推符号公式"""
    X_sample = np.linspace(x_range[0], x_range[1], n_samples)
    y_sample = nn_model.forward(X_sample)

    reg = SymbolicRegressor(
        pop_size=pop_size,
        max_depth=max_depth,
        generations=generations,
        random_state=random_state,
        verbose=verbose,
    )
    reg.fit(X_sample, y_sample)
    return reg, X_sample, y_sample


# ============================================================================
# 单个示例运行
# ============================================================================

def run_example(title, target_fn, x_range, nn_kwargs=None, train_kwargs=None,
                sr_kwargs=None, is_2d=False, X_2d=None, y_2d=None):
    """运行一个完整的 "黑箱→白箱" 示例
    
    返回: (target_expr, discovered_expr, r2, nn_mse, sym_mse, nn_params, elapsed)
    """
    nn_kwargs = nn_kwargs or {}
    train_kwargs = train_kwargs or {}
    sr_kwargs = sr_kwargs or {}

    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

    if is_2d:
        # 多变量：直接用原始数据（跳过 NN，数据本身就是"黑箱"）
        X_train, y_train = X_2d, y_2d
        X_eval, y_eval = X_2d, y_2d
        nn = None  # 无 NN，直接符号回归
    else:
        X_all = np.linspace(x_range[0], x_range[1], 300)
        y_all = target_fn(X_all)
        # 训练/评估数据分离
        idx = np.random.permutation(len(X_all))
        split = int(len(X_all) * 0.7)
        X_train, y_train = X_all[idx[:split]], y_all[idx[:split]]
        X_eval, y_eval = X_all[idx[split:]], y_all[idx[split:]]
        nn = TinyNN(input_dim=1, **nn_kwargs)

    # 训练神经网络（仅 1D 模式）
    if nn is not None:
        _train = {'epochs': 8000, 'lr': 0.02, 'decay': 0.9995, 'verbose': False}
        _train.update(train_kwargs)
        losses = nn.train(X_train, y_train, **_train)

    # 符号回归
    t0 = time.time()
    if is_2d:
        reg = SymbolicRegressor(**sr_kwargs)
        reg.fit(X_train, y_train)
    else:
        reg, X_sample, y_sample = explain_blackbox(nn, x_range, **sr_kwargs)
    elapsed = time.time() - t0

    # 评估
    discovered = reg.expression()
    r2 = reg.score(X_eval, y_eval)

    y_sym = reg.predict(X_eval)
    sym_mse = float(np.mean((y_sym - y_eval) ** 2))

    if nn is not None:
        y_nn = nn.forward(X_eval)
        nn_mse = float(np.mean((y_nn - y_eval) ** 2))
        nn_params = nn.param_count()
    else:
        nn_mse = float('nan')
        nn_params = 0

    # 简化表达式
    simplified = reg.best_genome_.simplify().to_expression()

    print(f"  表达式发现: {discovered}")
    if simplified != discovered:
        print(f"  简化后:     {simplified}")
    print(f"  R²:         {r2:.4f}")
    if nn is not None:
        print(f"  NN MSE:     {nn_mse:.6f}  |  Sym MSE: {sym_mse:.6f}")
    else:
        print(f"  Sym MSE:    {sym_mse:.6f}")
    print(f"  耗时:       {elapsed:.1f}s")

    return discovered, r2, nn_mse, sym_mse, nn_params, elapsed


# ============================================================================
# 主程序
# ============================================================================

def main():
    np.random.seed(42)

    print("=" * 60)
    print("  NeoGlyph 可解释 AI Demo v2")
    print("  黑箱 → 白箱：从神经网络中提取数学公式")
    print("=" * 60)

    results = []

    # ---- Example 1: 简单二次函数 ----
    discovered, r2, nn_mse, sym_mse, nn_params, t = run_example(
        title="Example 1: 简单二次函数",
        target_fn=lambda x: x**2 + 2*x + 1,
        x_range=(-4, 4),
        sr_kwargs={'pop_size': 80, 'max_depth': 3, 'generations': 200},
    )
    results.append(("f(x)=x²+2x+1", discovered, r2, nn_mse, sym_mse, nn_params, t))

    # ---- Example 2: 三角函数 ----
    discovered, r2, nn_mse, sym_mse, nn_params, t = run_example(
        title="Example 2: 三角函数组合",
        target_fn=lambda x: x + np.sin(x),
        x_range=(-3, 3),
        sr_kwargs={'pop_size': 80, 'max_depth': 4, 'generations': 200},
    )
    results.append(("f(x)=x+sin(x)", discovered, r2, nn_mse, sym_mse, nn_params, t))

    # ---- Example 3: 复杂函数 ----
    discovered, r2, nn_mse, sym_mse, nn_params, t = run_example(
        title="Example 3: 复杂嵌套函数（当前极限）",
        target_fn=lambda x: x**2 + np.sin(3*x) + 2,
        x_range=(-3, 3),
        nn_kwargs={'hidden_size': 20},
        sr_kwargs={'pop_size': 100, 'max_depth': 5, 'generations': 300},
    )
    results.append(("f(x)=x²+sin(3x)+2", discovered, r2, nn_mse, sym_mse, nn_params, t))

    # ---- Example 4: 多变量 ----
    X_2d = np.array([[i, j] for i in range(-5, 6) for j in range(-5, 6)], dtype=np.float64)
    y_2d = X_2d[:, 0] + X_2d[:, 1]  # f(x,y) = x + y

    discovered, r2, nn_mse, sym_mse, nn_params, t = run_example(
        title="Example 4: 多变量 f(x,y) = x + y（直接符号回归）",
        target_fn=None,
        x_range=None,
        is_2d=True, X_2d=X_2d, y_2d=y_2d,
        sr_kwargs={'pop_size': 100, 'max_depth': 3, 'generations': 200, 'random_state': 42},
    )
    results.append(("f(x,y)=x+y", discovered, r2, nn_mse, sym_mse, nn_params, t))

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    print("  汇总")
    print("=" * 60)
    print(f"  {'目标公式':<20s} {'R²':>6s}  {'NN MSE':>10s}  {'Sym MSE':>10s}  {'耗时':>6s}")
    print(f"  {'─'*20} {'─'*6} {'─'*10} {'─'*10} {'─'*6}")
    for target, expr, r2, nn_mse, sym_mse, nn_params, t in results:
        nn_str = f"{nn_mse:10.6f}" if not np.isnan(nn_mse) else "       N/A"
        print(f"  {target:<20s} {r2:6.4f}  {nn_str}  {sym_mse:10.6f}  {t:5.1f}s")

    # ---- 可解释性对比 ----
    print("\n" + "=" * 60)
    print("  可解释性对比")
    print("=" * 60)
    print(f"""
  神经网络（黑箱）:
    - 参数量:      {nn_params} 个浮点数
    - 可解释性:    无法用公式表达
    - 人类可读:    否
    - 可微分:      是
    - 可推导:      否

  符号公式（白箱）:
    - 表达式:      {results[0][1]}  (示例1)
    - 可解释性:    完全透明
    - 人类可读:    是
    - 可微分:      是（通过 VM 自动求导）
    - 可推导:      是（可用于证明、简化、教学）
""")

    print("=" * 60)
    print("  Demo 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()