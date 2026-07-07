"""
NeoGlyph 可视化模块
==================

提供拟合曲线、进化损失曲线、表达式树可视化等绘图功能。
matplotlib 为可选依赖，未安装时函数会提示。
"""

import numpy as np


def _check_matplotlib():
    """检查 matplotlib 是否可用"""
    try:
        import matplotlib
        return True
    except ImportError:
        return False


def plot_fit_curve(regressor, X, y, title="Fit Curve", save_path=None):
    """绘制拟合曲线：真实值 vs 预测值
    
    Parameters
    ----------
    regressor : SymbolicRegressor
        已拟合的回归器
    X : array-like
        输入数据
    y : array-like
        真实目标值
    title : str
        图表标题
    save_path : str or None
        保存路径，None 则显示
        
    Returns
    -------
    fig : matplotlib Figure or None
    """
    if not _check_matplotlib():
        print("需要 matplotlib: pip install matplotlib")
        return None
    
    import matplotlib.pyplot as plt
    
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    
    # 生成密集预测曲线
    if X.ndim == 1:
        X_smooth = np.linspace(X.min(), X.max(), 200)
        y_pred_smooth = regressor.predict(X_smooth)
    else:
        X_smooth = X
        y_pred_smooth = regressor.predict(X)
    
    y_pred = regressor.predict(X)
    r2 = regressor.score(X, y)
    expr = regressor.expression()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if X.ndim == 1:
        ax.scatter(X, y, alpha=0.6, label='Data', s=40, edgecolors='k', linewidth=0.5)
        ax.plot(X_smooth, y_pred_smooth, 'r-', linewidth=2, label=f'NeoGlyph: {expr}')
    else:
        ax.scatter(y, y_pred, alpha=0.6, label='Predictions', s=40, edgecolors='k', linewidth=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect')
    
    ax.set_xlabel('Input', fontsize=12)
    ax.set_ylabel('Output', fontsize=12)
    ax.set_title(f'{title}\nR² = {r2:.4f} | {expr}', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_evolution_history(history, title="Evolution Progress", save_path=None):
    """绘制进化损失曲线：每代最佳/平均 fitness
    
    Parameters
    ----------
    history : list[dict]
        regressor.history_ 或 engine.history
    title : str
        图表标题
    save_path : str or None
        保存路径
        
    Returns
    -------
    fig : matplotlib Figure or None
    """
    if not _check_matplotlib():
        print("需要 matplotlib: pip install matplotlib")
        return None
    
    import matplotlib.pyplot as plt
    
    gens = [h.get('generation', i+1) for i, h in enumerate(history)]
    best = [h.get('best_fitness', 0) for h in history]
    avg = [h.get('avg_fitness', 0) for h in history]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(gens, best, 'b-', linewidth=2, label='Best Fitness')
    ax.plot(gens, avg, 'orange', alpha=0.7, linewidth=1.5, label='Average Fitness')
    ax.fill_between(gens, avg, best, alpha=0.15, color='blue')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, label='Perfect (1.0)')
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_expression_tree(genome, title="Expression Tree", save_path=None):
    """绘制表达式树结构
    
    Parameters
    ----------
    genome : TreeGenome
        表达式树
    title : str
        图表标题
    save_path : str or None
        保存路径
        
    Returns
    -------
    fig : matplotlib Figure or None
    """
    if not _check_matplotlib():
        print("需要 matplotlib: pip install matplotlib")
        return None
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    if genome.root is None:
        return None
    
    def _get_tree_layout(node, x=0, y=0, level=0, positions=None, edges=None):
        if positions is None:
            positions = {}
        if edges is None:
            edges = []
        
        positions[id(node)] = (x, y, node)
        
        if node.node_type == 'operation':
            if node.left:
                edges.append((id(node), id(node.left)))
                _get_tree_layout(node.left, x - 1.0 / (2**level), y - 1, level + 1, positions, edges)
            if node.right:
                edges.append((id(node), id(node.right)))
                _get_tree_layout(node.right, x + 1.0 / (2**level), y - 1, level + 1, positions, edges)
        
        return positions, edges
    
    positions, edges = _get_tree_layout(genome.root)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-genome.root.get_depth() - 0.5, 1.0)
    ax.axis('off')
    ax.set_title(title, fontsize=13, pad=15)
    
    # 颜色映射
    colors = {
        'operation': '#4A90D9',
        'variable': '#50C878',
        'constant': '#F5A623',
    }
    
    # 画边
    for parent_id, child_id in edges:
        px, py, _ = positions[parent_id]
        cx, cy, _ = positions[child_id]
        ax.plot([px, cx], [py, cy], 'k-', alpha=0.3, linewidth=1.5)
    
    # 画节点
    for node_id, (x, y, node) in positions.items():
        color = colors.get(node.node_type, '#999')
        circle = patches.Circle((x, y), 0.25, facecolor=color, edgecolor='white', 
                                linewidth=2, zorder=3)
        ax.add_patch(circle)
        
        if node.node_type == 'constant':
            label = f"{node.value:.1f}"
        elif node.node_type == 'variable':
            label = node.name
        else:
            label = node.op
        
        ax.text(x, y, label, ha='center', va='center', fontsize=9, 
                fontweight='bold', color='white', zorder=4)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def print_tree(genome):
    """在终端打印表达式树（ASCII）
    
    Parameters
    ----------
    genome : TreeGenome
        表达式树
    """
    def _print_node(node, prefix="", is_last=True):
        if node is None:
            return
        
        connector = "└── " if is_last else "├── "
        if node.node_type == 'constant':
            label = f"Const({node.value:.2f})"
        elif node.node_type == 'variable':
            label = f"Var({node.name})"
        else:
            label = node.op
        
        print(f"{prefix}{connector}{label}")
        
        if node.node_type == 'operation':
            new_prefix = prefix + ("    " if is_last else "│   ")
            children = []
            if node.left:
                children.append(node.left)
            if node.right:
                children.append(node.right)
            for i, child in enumerate(children):
                _print_node(child, new_prefix, i == len(children) - 1)
    
    if genome.root is None:
        print("(empty tree)")
        return
    
    label = "EXPR"
    print(f"└── {label}")
    _print_node(genome.root, "", True)


def plot_pareto_front(pareto_genomes, save_path=None):
    """绘制帕累托前沿：误差 vs 复杂度
    
    Parameters
    ----------
    pareto_genomes : list[dict]
        每项包含 mse, complexity, expression
    save_path : str or None
        保存路径
        
    Returns
    -------
    fig : matplotlib Figure or None
    """
    if not _check_matplotlib():
        print("需要 matplotlib: pip install matplotlib")
        return None
    
    import matplotlib.pyplot as plt
    
    mse_vals = [g['mse'] for g in pareto_genomes]
    complexity_vals = [g['complexity'] for g in pareto_genomes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(complexity_vals, mse_vals, s=80, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    # 标注最佳的 3 个
    sorted_by_mse = sorted(pareto_genomes, key=lambda g: g['mse'])
    for g in sorted_by_mse[:3]:
        ax.annotate(g['expression'], (g['complexity'], g['mse']),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=9, ha='center')
    
    ax.set_xlabel('Complexity (nodes)', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Pareto Front: Accuracy vs Simplicity', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig