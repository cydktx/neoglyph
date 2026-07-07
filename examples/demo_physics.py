"""
NeoGlyph 应用 Demo: 物理公式发现
================================

从实验数据中自动发现物理定律。

演示场景：
1. 自由落体位移公式: s = 0.5 * g * t²
2. 弹簧胡克定律: F = k * x
3. 动能公式: E = 0.5 * m * v²
4. 简单线性回归: y = ax + b
"""

import numpy as np
from neoglyph.applications import SymbolicRegressor, PhysicsDiscoverer


def demo_free_fall():
    """Demo 1: 自由落体位移公式 s = ½gt²"""
    print("=" * 60)
    print("Demo 1: 自由落体位移公式发现")
    print("目标公式: s = 0.5 * g * t²  (g ≈ 9.8)")
    print("=" * 60)

    g = 9.8
    t = np.linspace(0.5, 5.0, 15)
    s = 0.5 * g * t ** 2

    noise = np.random.normal(0, 0.5, size=len(t))
    s_noisy = s + noise

    print(f"数据点: {len(t)} 个")
    print(f"时间范围: [{t[0]:.1f}, {t[-1]:.1f}] s")
    print(f"位移范围: [{s_noisy.min():.1f}, {s_noisy.max():.1f}] m")
    print()

    discoverer = PhysicsDiscoverer(
        pop_size=100,
        max_depth=4,
        generations=150,
        mdl_weight=0.005,
        verbose=False,
        random_state=42,
    )

    result = discoverer.discover(t, s_noisy, variable_name='t')

    print(f"发现公式: {result['expression']}")
    print(f"R² 分数: {result['r2_score']:.4f}")
    print(f"MSE: {result['mse']:.4f}")
    print(f"MAE: {result['mae']:.4f}")
    print(f"表达式复杂度: {result['complexity']}")

    passed = result['r2_score'] > 0.8
    if passed:
        print("\n✅ 成功发现自由落体二次关系！")
    else:
        print("\n⚠️  未完全收敛，可增加代数或种群大小")

    print()
    return result['r2_score']


def demo_hookes_law():
    """Demo 2: 胡克定律 F = k * x"""
    print("=" * 60)
    print("Demo 2: 胡克定律发现")
    print("目标公式: F = k * x  (k = 20 N/m)")
    print("=" * 60)

    k = 20.0
    x = np.linspace(0.05, 0.5, 15)
    F = k * x

    noise = np.random.normal(0, 0.3, size=len(x))
    F_noisy = F + noise

    print(f"数据点: {len(x)} 个")
    print(f"伸长量范围: [{x.min():.2f}, {x.max():.2f}] m")
    print(f"弹力范围: [{F_noisy.min():.1f}, {F_noisy.max():.1f}] N")
    print()

    reg = SymbolicRegressor(
        pop_size=60,
        max_depth=2,
        generations=80,
        verbose=False,
        random_state=42,
    )
    reg.fit(x, F_noisy)

    r2 = reg.score(x, F_noisy)
    print(f"发现公式: {reg.best_expression()}")
    print(f"R² 分数: {r2:.4f}")

    passed = r2 > 0.9
    if passed:
        print("\n✅ 成功发现胡克定律（线性关系）！")
    else:
        print("\n⚠️  未完全收敛")

    print()
    return r2


def demo_kinetic_energy():
    """Demo 3: 动能公式 E = ½mv²"""
    print("=" * 60)
    print("Demo 3: 动能公式发现")
    print("目标公式: E = 0.5 * m * v²  (m = 2 kg)")
    print("=" * 60)

    m = 2.0
    v = np.linspace(1.0, 10.0, 15)
    E = 0.5 * m * v ** 2

    noise = np.random.normal(0, 2.0, size=len(v))
    E_noisy = E + noise
    E_noisy = np.maximum(E_noisy, 0.1)

    print(f"数据点: {len(v)} 个")
    print(f"速度范围: [{v.min():.1f}, {v.max():.1f}] m/s")
    print(f"动能范围: [{E_noisy.min():.1f}, {E_noisy.max():.1f}] J")
    print()

    discoverer = PhysicsDiscoverer(
        pop_size=100,
        max_depth=4,
        generations=150,
        mdl_weight=0.005,
        verbose=False,
        random_state=42,
    )

    result = discoverer.discover(v, E_noisy, variable_name='v')

    print(f"发现公式: {result['expression']}")
    print(f"R² 分数: {result['r2_score']:.4f}")
    print(f"表达式复杂度: {result['complexity']}")

    passed = result['r2_score'] > 0.9
    if passed:
        print("\n✅ 成功发现动能公式结构（v² 关系）！")
    else:
        print("\n⚠️  未完全收敛")

    print()
    return result['r2_score']


def demo_simple_linear():
    """Demo 4: 简单线性回归 y = ax + b"""
    print("=" * 60)
    print("Demo 4: 简单线性回归")
    print("目标公式: y = 2x + 1")
    print("=" * 60)

    x = np.linspace(-5, 5, 20)
    y = 2 * x + 1

    reg = SymbolicRegressor(
        pop_size=50,
        max_depth=2,
        generations=80,
        verbose=False,
        random_state=42,
    )
    reg.fit(x, y)

    r2 = reg.score(x, y)
    print(f"发现公式: {reg.best_expression()}")
    print(f"R² 分数: {r2:.4f}")

    passed = r2 > 0.99
    if passed:
        print("\n✅ 完美拟合线性函数！")
    else:
        print("\n⚠️  未完全收敛")

    print()
    return r2


def main():
    """运行所有 Demo"""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " NeoGlyph 应用层 Demo: 物理公式发现".center(58) + "║")
    print("║" + " Physics Formula Discovery with Symbolic Regression".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    results = []

    try:
        score = demo_free_fall()
        results.append(('自由落体', score))
    except Exception as e:
        print(f"Demo 1 失败: {e}\n")
        results.append(('自由落体', 0.0))

    try:
        score = demo_hookes_law()
        results.append(('胡克定律', score))
    except Exception as e:
        print(f"Demo 2 失败: {e}\n")
        results.append(('胡克定律', 0.0))

    try:
        score = demo_kinetic_energy()
        results.append(('动能公式', score))
    except Exception as e:
        print(f"Demo 3 失败: {e}\n")
        results.append(('动能公式', 0.0))

    try:
        score = demo_simple_linear()
        results.append(('线性回归', score))
    except Exception as e:
        print(f"Demo 4 失败: {e}\n")
        results.append(('线性回归', 0.0))

    print("=" * 60)
    print("总结")
    print("=" * 60)
    print(f"{'Demo':<15} {'R² Score':<12} {'状态'}")
    print("-" * 60)
    all_pass = True
    for name, score in results:
        status = "✅ 通过" if score > 0.8 else "⚠️  未达标"
        print(f"{name:<15} {score:<12.4f} {status}")
        if score <= 0.8:
            all_pass = False
    print("-" * 60)
    if all_pass:
        print("🎉 所有 Demo 全部通过！")
    else:
        print("部分 Demo 未达标，可尝试增加种群大小或进化代数。")
    print()


if __name__ == "__main__":
    main()
