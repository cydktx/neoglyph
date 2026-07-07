#!/usr/bin/env python3
"""
Debug Tree Genome - 测试基本功能
"""

import numpy as np
from neoglyph import TreeGenome, ConstantNode, VariableNode, OperationNode


def test_basic_tree():
    """测试基本Tree结构"""
    print("\n" + "="*60)
    print("  Basic Tree Genome Test")
    print("="*60)
    
    # 手动构建 y = 2*x + 1
    print("\n1. Manual construction: y = 2*x + 1")
    
    # 构建树结构
    x_node = VariableNode('x')
    two_node = ConstantNode(2.0)
    mul_node = OperationNode('MUL', x_node, two_node)  # x * 2
    
    one_node = ConstantNode(1.0)
    add_node = OperationNode('ADD', mul_node, one_node)  # (x*2) + 1
    
    genome = TreeGenome(add_node)
    
    print(f"Expression: {genome.to_expression()}")
    print(f"VM Code:\n{genome.to_vm_code()}")
    
    # 测试评估
    print("\nEvaluation:")
    for x in [-5, -3, -1, 1, 3, 5]:
        result = genome.root.evaluate(x)
        target = x * 2 + 1
        error = abs(result - target)
        print(f"x={x}: result={result:.4f}, target={target:.4f}, error={error:.4f}")
    
    # 计算fitness
    inputs = np.array([-5, -3, -1, 1, 3, 5], dtype=np.float32)
    target_fn = lambda x: x * 2 + 1
    
    fitness = genome.calculate_fitness(inputs, target_fn)
    print(f"\nFitness: {fitness:.4f}")
    print(f"MDL Score: {genome.mdl_score:.2f}")
    print(f"Complexity: {genome.get_complexity()}")


def test_random_tree():
    """测试随机Tree Genome"""
    print("\n" + "="*60)
    print("  Random Tree Genome Test")
    print("="*60)
    
    target_fn = lambda x: x * 2 + 1
    inputs = np.array([-5, -3, -1, 1, 3, 5], dtype=np.float32)
    
    print("\n2. Testing 10 random Tree Genomes")
    
    for i in range(10):
        genome = TreeGenome.create_random(max_depth=2)
        
        print(f"\nGenome {i+1}:")
        print(f"Expression: {genome.to_expression()}")
        
        # 测试评估
        eval_result = genome.evaluate_with_target(inputs, target_fn)
        
        print(f"Valid Count: {eval_result['valid_count']}")
        print(f"MSE: {eval_result['mse']:.4f}")
        print(f"Accuracy: {eval_result['accuracy']:.4f}")
        
        # 计算fitness
        fitness = genome.calculate_fitness(inputs, target_fn)
        print(f"Fitness: {fitness:.4f}")
        
        # 示例评估
        try:
            result = genome.root.evaluate(1.0)
            print(f"Example: x=1 -> {result:.4f}, target={target_fn(1)}")
        except Exception as e:
            print(f"Error evaluating x=1: {e}")


def test_node_types():
    """测试节点类型"""
    print("\n" + "="*60)
    print("  Node Type Test")
    print("="*60)
    
    # 测试各种节点
    print("\n3. Testing node types")
    
    # ConstantNode
    c_node = ConstantNode(3.5)
    print(f"ConstantNode(3.5): evaluate(10) = {c_node.evaluate(10)}")
    
    # VariableNode
    v_node = VariableNode('x')
    print(f"VariableNode('x'): evaluate(10) = {v_node.evaluate(10)}")
    
    # OperationNode
    add_node = OperationNode('ADD', c_node, v_node)
    print(f"ADD(3.5, x): evaluate(10) = {add_node.evaluate(10)}")
    
    mul_node = OperationNode('MUL', v_node, c_node)
    print(f"MUL(x, 3.5): evaluate(10) = {mul_node.evaluate(10)}")


if __name__ == '__main__':
    test_node_types()
    test_basic_tree()
    test_random_tree()