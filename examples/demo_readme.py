"""README 演示脚本 - 生成终端截图用的输出"""
import sys
sys.path.insert(0, '.')
import numpy as np
from neoglyph import TreeGenome, ConstantNode, VariableNode, OperationNode
from neoglyph import SymbolicRegressor
from neoglyph.evolution import EvolutionEngine

def demo_manual_tree():
    """演示1：手动构建表达式树"""
    print("=" * 60)
    print("  演示 1：手动构建表达式树")
    print("=" * 60)
    x = VariableNode('x')
    mul = OperationNode('MUL', x, ConstantNode(2.0))
    add = OperationNode('ADD', mul, ConstantNode(1.0))
    tree = TreeGenome(add)
    
    print(f"  表达式: {tree.to_expression()}")
    for val in [-3, 0, 5]:
        print(f"  f({val:>2}) = {tree.root.evaluate(val):.0f}")
    print()

def demo_symbolic_simplify():
    """演示2：符号简化"""
    print("=" * 60)
    print("  演示 2：符号简化器")
    print("=" * 60)
    
    examples = [
        ("3*x + 2*x + 1",
         OperationNode('ADD',
             OperationNode('ADD',
                 OperationNode('MUL', ConstantNode(3.0), VariableNode('x')),
                 OperationNode('MUL', ConstantNode(2.0), VariableNode('x'))),
             ConstantNode(1.0))),
        ("x + x + x + 5",
         OperationNode('ADD',
             OperationNode('ADD',
                 OperationNode('ADD', VariableNode('x'), VariableNode('x')),
                 VariableNode('x')),
             ConstantNode(5.0))),
        ("sin(0) + 2*1",
         OperationNode('ADD',
             OperationNode('SIN', ConstantNode(0.0)),
             OperationNode('MUL', ConstantNode(2.0), ConstantNode(1.0)))),
    ]
    
    for desc, node in examples:
        tree = TreeGenome(node)
        simplified = tree.simplify()
        print(f"  {desc:<20} →  {simplified.to_expression()}")
    print()

def demo_evolution():
    """演示3：自动发现公式"""
    print("=" * 60)
    print("  演示 3：自动发现公式 f(x) = 2x + 1")
    print("=" * 60)
    
    X = np.array([-5, -3, -1, 0, 1, 3, 5], dtype=np.float64)
    target_fn = lambda x: 2 * x + 1
    
    import random
    random.seed(123)
    np.random.seed(123)
    
    pop = [TreeGenome.create_random(max_depth=2) for _ in range(100)]
    best_fitness = 0
    
    for gen in range(150):
        for g in pop:
            g.calculate_fitness(X, target_fn)
        pop.sort(key=lambda g: g.fitness, reverse=True)
        if pop[0].fitness > best_fitness:
            best_fitness = pop[0].fitness
        if gen in [0, 30, 60, 90, 120, 149]:
            expr = pop[0].to_expression()
            print(f"  Gen {gen:3d}: fitness={pop[0].fitness:.4f}  expr={expr}")
        elite = [g.copy() for g in pop[:10]]
        new_pop = elite[:]
        while len(new_pop) < 100:
            p1 = random.choice(pop[:50])
            p2 = random.choice(pop[:50])
            child = TreeGenome.crossover(p1, p2)
            child.mutate(0.3, fitness=p1.fitness)
            new_pop.append(child)
        pop = new_pop
    
    best = pop[0].simplify()
    final = best.to_expression()
    preds = np.array([best.root.evaluate(x) for x in X])
    expected = np.array([target_fn(x) for x in X])
    mse = np.mean((preds - expected) ** 2)
    print(f"\n  最终发现: {final}")
    print(f"  MSE: {mse:.8f}")
    print()

def demo_multi_variable():
    """演示4：多变量回归"""
    print("=" * 60)
    print("  演示 4：多变量符号回归 f(x,y) = x + y")
    print("=" * 60)
    
    X = np.array([[i, j] for i in range(-3, 4) for j in range(-3, 4)], dtype=np.float64)
    y = X[:, 0] + X[:, 1]
    
    reg = SymbolicRegressor(
        pop_size=80, max_depth=2, generations=100,
        random_state=42, verbose=False,
    )
    reg.fit(X, y)
    
    expr = reg.expression()
    r2 = reg.score(X, y)
    print(f"  发现表达式: {expr}")
    print(f"  R² = {r2:.4f}")
    print()

def demo_vm_code():
    """演示5：VM 代码编译"""
    print("=" * 60)
    print("  演示 5：表达式 → VM 指令")
    print("=" * 60)
    
    tree = TreeGenome(OperationNode('ADD',
        OperationNode('MUL', VariableNode('x'), ConstantNode(2.0)),
        ConstantNode(1.0)))
    
    print(f"  表达式: {tree.to_expression()}")
    print(f"  VM 代码:")
    for line in tree.to_vm_code().split('\n'):
        print(f"    {line}")
    print()

if __name__ == '__main__':
    demo_manual_tree()
    demo_symbolic_simplify()
    demo_evolution()
    demo_multi_variable()
    demo_vm_code()
    print("=" * 60)
    print("  全部演示完成。")
    print("=" * 60)