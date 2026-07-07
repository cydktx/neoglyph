"""
NeoGlyph 应用 Demo: 数学函数逼近
===================================

从数据中自动发现数学函数表达式。

演示场景：
1. 二次函数: y = x² + 2x + 1
2. 三角函数组合: y = sin(x) + cos(x)
3. 指数衰减: y = exp(-x/2)
4. 三次多项式: y = x³ - 3x + 2
"""

import numpy as np
from neoglyph.applications import SymbolicRegressor


def demo_quadratic():
    """Demo 1: 二次函数 y = x² + 2x + 1 = (x+1)²"""
    print("=" * 60)
    print("Demo 1: 二次函数发现")
    print("目标函数: y = x² + 2x + 1")
    print("=" * 60)

    x = np.linspace(-4, 4, 20)
    y = x ** 2 + 2 * x + 1

    print(f"数据点: {len(x)} 个")
    print(f"x 范围: [{x.min():.1f}, {x.max():.1f}]")
    print(f"y 范围: [{y.min():.1f}, {y.max():.1f}]")
    print()

    reg = SymbolicRegressor(
        pop_size=80,
        max_depth=3,
        generations=150,
        mdl_weight=0.01,
        verbose=False,
        random_state=42,
    )
    reg.fit(x, y)

    r2 = reg.score(x, y)
    print(f"发现表达式: {reg.best_expression()}")
    print(f"R² 分数: {r2:.4f}")

    passed = r2 > 0.9
    if passed:
        print("\n✅ 成功发现二次函数结构！")
    else:
        print("\n⚠️  未完全收敛")
    print()
    return r2


def demo_sin_cos():
    """Demo 2: 三角函数组合 y = sin(x) + cos(x)"""
    print("=" * 60)
    print("Demo 2: 三角函数组合发现")
    print("目标函数: y = sin(x) + cos(x)")
    print("=" * 60)

    x = np.linspace(0, 2 * np.pi, 20)
    y = np.sin(x) + np.cos(x)

    print(f"数据点: {len(x)} 个")
    print(f"x 范围: [{x.min():.2f}, {x.max():.2f}]")
    print(f"y 范围: [{y.min():.2f}, {y.max():.2f}]")
    print()

    reg = SymbolicRegressor(
        pop_size=100,
        max_depth=3,
        generations=200,
        mdl_weight=0.005,
        verbose=False,
        random_state=42,
    )
    reg.fit(x, y)

    r2 = reg.score(x, y)
    print(f"发现表达式: {reg.best_expression()}")
    print(f"R² 分数: {r2:.4f}")

    passed = r2 > 0.5
    if passed:
        print("\n✅ 发现了三角函数组合结构！")
    else:
        print("\n⚠️  三角函数较难，建议增加种群或代数")
    print()
    return r2


def demo_exp_decay():
    """Demo 3: 指数衰减 y = exp(-x/2)"""
    print("=" * 60)
    print("Demo 3: 指数衰减发现")
    print("目标函数: y = exp(-x/2)")
    print("=" * 60)

    x = np.linspace(0.1, 3.0, 15)
    y = np.exp(-x / 2)

    print(f"数据点: {len(x)} 个")
    print(f"x 范围: [{x.min():.1f}, {x.max():.1f}]")
    print(f"y 范围: [{y.min():.3f}, {y.max():.3f}]")
    print()

    reg = SymbolicRegressor(
        pop_size=80,
        max_depth=3,
        generations=150,
        mdl_weight=0.005,
        verbose=False,
        random_state=42,
    )
    reg.fit(x, y)

    r2 = reg.score(x, y)
    print(f"发现表达式: {reg.best_expression()}")
    print(f"R² 分数: {r2:.4f}")

    passed = r2 > 0.7
    if passed:
        print("\n✅ 成功发现指数衰减关系！")
    else:
        print("\n⚠️  未完全收敛")
    print()
    return r2


def demo_cubic():
    """Demo 4: 三次多项式 y = x³ - 3x + 2"""
    print("=" * 60)
    print("Demo 4: 三次多项式发现")
    print("目标函数: y = x³ - 3x + 2")
    print("=" * 60)

    x = np.linspace(-2, 3, 20)
    y = x ** 3 - 3 * x + 2

    print(f"数据点: {len(x)} 个")
    print(f"x 范围: [{x.min():.1f}, {x.max():.1f}]")
    print(f"y 范围: [{y.min():.1f}, {y.max():.1f}]")
    print()

    reg = SymbolicRegressor(
        pop_size=120,
        max_depth=4,
        generations=200,
        mdl_weight=0.005,
        verbose=False,
        random_state=42,
    )
    reg.fit(x, y)

    r2 = reg.score(x, y)
    print(f"发现表达式: {reg.best_expression()}")
    print(f"R² 分数: {r2:.4f}")

    passed = r2 > 0.8
    if passed:
        print("\n✅ 成功发现三次多项式结构！")
    else:
        print("\n⚠️  三次函数较复杂，建议增加种群或代数")
    print()
    return r2


def main():
    """运行所有 Demo"""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " NeoGlyph 应用层 Demo: 数学函数逼近".center(58) + "║")
    print("║" + "  Mathematical Function Discovery".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    demos = [
        ('二次函数', demo_quadratic),
        ('三角函数', demo_sin_cos),
        ('指数衰减', demo_exp_decay),
        ('三次多项式', demo_cubic),
    ]

    results = []
    for name, fn in demos:
        try:
            score = fn()
            results.append((name, score))
        except Exception as e:
            print(f"{name} Demo 失败: {e}\n")
            results.append((name, 0.0))

    print("=" * 60)
    print("总结")
    print("=" * 60)
    print(f"{'函数':<15} {'R² Score':<12} {'状态'}")
    print("-" * 60)
    all_pass = True
    for name, score in results:
        status = "✅ 通过" if score > 0.7 else "⚠️  挑战"
        print(f"{name:<15} {score:<12.4f} {status}")
        if score <= 0.7:
            all_pass = False
    print("-" * 60)
    if all_pass:
        print("🎉 所有函数发现任务全部通过！")
    else:
        print("部分任务较难（三角函数、三次函数），可增加种群规模和进化代数。")
    print()


if __name__ == "__main__":
    main()
