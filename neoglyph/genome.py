import numpy as np
import random
import copy
from .vm import NeoGlyphVM


# ============================================================================
# Tree Genome - Evolution Core 3.0
# ============================================================================

class ProgramNode:
    """程序节点基类"""
    
    def __init__(self):
        self.parent = None
        self.protected = False  # Protected Gene标记
        self.fitness_contribution = 0.0
    
    def evaluate(self, x):
        """计算节点值"""
        raise NotImplementedError
    
    def to_expression(self):
        """转换为可读表达式"""
        raise NotImplementedError
    
    def to_vm_code(self):
        """转换为VM代码"""
        raise NotImplementedError
    
    def copy(self):
        """深拷贝节点"""
        raise NotImplementedError
    
    def get_depth(self):
        """获取节点深度"""
        raise NotImplementedError
    
    def get_size(self):
        """获取节点大小（子节点数量）"""
        raise NotImplementedError
    
    def get_nodes(self):
        """获取所有节点列表"""
        raise NotImplementedError


class ConstantNode(ProgramNode):
    """常数节点 - Evolution Core 3.1增强"""
    
    def __init__(self, value):
        super().__init__()
        self.value = float(value)
        self.node_type = 'constant'
    
    def evaluate(self, x):
        return self.value
    
    def to_expression(self):
        """转换为可读表达式 - Evolution Core 3.1改进"""
        # 规范化显示：整数不显示小数
        if self.value == round(self.value):
            display_val = int(round(self.value))
        else:
            display_val = self.value
        
        if display_val < 0:
            return f"({display_val})"
        return str(display_val)
    
    def to_vm_code(self):
        return f"PUSH {self.value}"
    
    def copy(self):
        new_node = ConstantNode(self.value)
        new_node.protected = self.protected
        new_node.fitness_contribution = self.fitness_contribution
        return new_node
    
    def get_depth(self):
        return 0
    
    def get_size(self):
        return 1
    
    def get_nodes(self):
        return [self]
    
    def __repr__(self):
        return f"Const({self.value:.2f})"


class VariableNode(ProgramNode):
    """变量节点"""
    
    def __init__(self, name='x'):
        super().__init__()
        self.name = name
        self.node_type = 'variable'
    
    def evaluate(self, x):
        return float(x)
    
    def to_expression(self):
        return self.name
    
    def to_vm_code(self):
        var_map = {'x': 'a'}
        vm_var = var_map.get(self.name, self.name)
        return f"LOAD {vm_var}"
    
    def copy(self):
        new_node = VariableNode(self.name)
        new_node.protected = self.protected
        new_node.fitness_contribution = self.fitness_contribution
        return new_node
    
    def get_depth(self):
        return 0
    
    def get_size(self):
        return 1
    
    def get_nodes(self):
        return [self]
    
    def __repr__(self):
        return f"Var({self.name})"


class OperationNode(ProgramNode):
    """操作节点 - Evolution Core 3.2增强
    
    新增：
    - 同类项合并
    - 多级表达式化简
    - 简化结果缓存
    """
    
    OPERATIONS = {
        'ADD': lambda a, b: a + b,
        'SUB': lambda a, b: a - b,
        'MUL': lambda a, b: a * b,
        'DIV': lambda a, b: a / b if b != 0 else 0.0,
    }
    
    COMMUTATIVE_OPS = ['ADD', 'MUL']
    
    def __init__(self, op, left=None, right=None):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right
        self.node_type = 'operation'
        self._simplified_cache = None
        self._cache_dirty = True
        
        if left:
            left.parent = self
        if right:
            right.parent = self
    
    def _mark_dirty(self):
        """标记缓存为脏"""
        self._cache_dirty = True
        self._simplified_cache = None
        if self.parent and hasattr(self.parent, '_mark_dirty'):
            self.parent._mark_dirty()
    
    def evaluate(self, x):
        if self.left is None or self.right is None:
            return 0.0
        
        left_val = self.left.evaluate(x)
        right_val = self.right.evaluate(x)
        
        try:
            return self.OPERATIONS[self.op](left_val, right_val)
        except Exception:
            return 0.0
    
    def simplify(self):
        """符号简化 - Evolution Core 3.2增强
        
        新增：
        - 同类项合并 (3*x + 2*x → 5*x)
        - 多级表达式化简
        - 简化结果缓存
        """
        if not self._cache_dirty and self._simplified_cache is not None:
            return self._simplified_cache
        
        if self.left is None or self.right is None:
            self._simplified_cache = self
            self._cache_dirty = False
            return self
        
        # 先简化子节点
        if self.left.node_type == 'operation':
            simplified_left = self.left.simplify()
            if simplified_left != self.left:
                self.left = simplified_left
                self.left.parent = self
        if self.right.node_type == 'operation':
            simplified_right = self.right.simplify()
            if simplified_right != self.right:
                self.right = simplified_right
                self.right.parent = self
        
        # 常数规范化
        self._canonicalize_constants()
        
        # 同类项收集与合并（ADD/SUB）
        if self.op in ['ADD', 'SUB']:
            merged = self._collect_and_merge_like_terms()
            if merged is not None:
                self._simplified_cache = merged
                self._cache_dirty = False
                return merged
        
        # 常数合并
        if self.left.node_type == 'constant' and self.right.node_type == 'constant':
            merged = self._merge_constants()
            self._simplified_cache = merged
            self._cache_dirty = False
            return merged
        
        # 基础简化规则
        simplified = self._apply_simplification_rules()
        if simplified:
            self._simplified_cache = simplified
            self._cache_dirty = False
            return simplified
        
        # 交换律规范化
        self._normalize_commutative()
        
        self._simplified_cache = self
        self._cache_dirty = False
        return self
    
    def _collect_and_merge_like_terms(self):
        """收集并合并同类项
        
        支持：
        - a*x + b*x → (a+b)*x
        - a*x - b*x → (a-b)*x
        - x + c*x → (1+c)*x
        - a*x + x → (a+1)*x
        - (a*x+b) + (c*x+d) → (a+c)*x + (b+d)
        """
        if self.op not in ['ADD', 'SUB']:
            return None
        
        left_info = self._extract_term_info(self.left)
        right_info = self._extract_term_info(self.right)
        
        if left_info is None or right_info is None:
            return None
        
        # 合并系数
        left_x_coeff, left_const = left_info
        right_x_coeff, right_const = right_info
        
        # 根据操作符调整右子树系数
        if self.op == 'SUB':
            right_x_coeff = -right_x_coeff
            right_const = -right_const
        
        total_x_coeff = left_x_coeff + right_x_coeff
        total_const = left_const + right_const
        
        # 规范化零值
        total_x_coeff = self._canonicalize_value(total_x_coeff)
        total_const = self._canonicalize_value(total_const)
        
        # 构建结果
        x_part = None
        const_part = None
        
        if total_x_coeff != 0:
            if total_x_coeff == 1:
                x_part = VariableNode('x')
            else:
                x_part = OperationNode('MUL', 
                                        VariableNode('x'), 
                                        ConstantNode(total_x_coeff))
        
        if total_const != 0:
            const_part = ConstantNode(total_const)
        
        # 组合结果
        if x_part is not None and const_part is not None:
            if total_const > 0:
                return OperationNode('ADD', x_part, const_part)
            else:
                return OperationNode('SUB', x_part, ConstantNode(abs(total_const)))
        elif x_part is not None:
            return x_part
        elif const_part is not None:
            return const_part
        else:
            return ConstantNode(0.0)
    
    def _extract_term_info(self, node):
        """提取节点的项信息 (x系数, 常数项)
        
        返回 (x_coefficient, constant_term) 或 None
        """
        if node.node_type == 'constant':
            return (0.0, node.value)
        
        if node.node_type == 'variable':
            return (1.0, 0.0)
        
        if node.node_type == 'operation':
            if node.op == 'MUL':
                # a*x 或 x*a
                if node.left.node_type == 'constant' and node.right.node_type == 'variable':
                    return (node.left.value, 0.0)
                if node.left.node_type == 'variable' and node.right.node_type == 'constant':
                    return (node.right.value, 0.0)
                # 两个都是常数
                if node.left.node_type == 'constant' and node.right.node_type == 'constant':
                    return (0.0, node.left.value * node.right.value)
            
            elif node.op == 'ADD':
                left_info = self._extract_term_info(node.left)
                right_info = self._extract_term_info(node.right)
                if left_info and right_info:
                    return (left_info[0] + right_info[0], left_info[1] + right_info[1])
            
            elif node.op == 'SUB':
                left_info = self._extract_term_info(node.left)
                right_info = self._extract_term_info(node.right)
                if left_info and right_info:
                    return (left_info[0] - right_info[0], left_info[1] - right_info[1])
        
        return None
    
    def _canonicalize_constants(self):
        """常数规范化：近似整数恢复成整数"""
        if self.left.node_type == 'constant':
            self.left.value = self._canonicalize_value(self.left.value)
        if self.right.node_type == 'constant':
            self.right.value = self._canonicalize_value(self.right.value)
    
    def _canonicalize_value(self, value):
        """规范化单个数值"""
        if abs(value - round(value)) < 0.01:
            return float(round(value))
        
        for denom in [2, 3, 4, 5]:
            frac = round(value * denom) / denom
            if abs(value - frac) < 0.01:
                return frac
        
        return value
    
    def _merge_constants(self):
        """合并常数节点"""
        left_val = self.left.value
        right_val = self.right.value
        
        try:
            result = self.OPERATIONS[self.op](left_val, right_val)
            result = self._canonicalize_value(result)
            return ConstantNode(result)
        except Exception:
            return self
    
    def _apply_simplification_rules(self):
        """应用简化规则 - Evolution Core 3.2增强"""
        left = self.left
        right = self.right
        
        # x + 0 => x
        if self.op == 'ADD':
            if right.node_type == 'constant' and right.value == 0.0:
                return left.copy()
            if left.node_type == 'constant' and left.value == 0.0:
                return right.copy()
        
        # x - 0 => x
        if self.op == 'SUB':
            if right.node_type == 'constant' and right.value == 0.0:
                return left.copy()
            if right.node_type == 'constant' and right.value < 0:
                new_const = ConstantNode(abs(right.value))
                return OperationNode('ADD', left.copy(), new_const)
        
        # x * 1 => x
        if self.op == 'MUL':
            if right.node_type == 'constant' and right.value == 1.0:
                return left.copy()
            if left.node_type == 'constant' and left.value == 1.0:
                return right.copy()
        
        # x * 0 => 0
        if self.op == 'MUL':
            if right.node_type == 'constant' and right.value == 0.0:
                return ConstantNode(0.0)
            if left.node_type == 'constant' and left.value == 0.0:
                return ConstantNode(0.0)
        
        # x / 1 => x
        if self.op == 'DIV':
            if right.node_type == 'constant' and right.value == 1.0:
                return left.copy()
        
        # x / x => 1
        if self.op == 'DIV':
            if (left.node_type == 'variable' and right.node_type == 'variable' and 
                left.name == right.name):
                return ConstantNode(1.0)
        
        # x + x => 2*x
        if self.op == 'ADD':
            if (left.node_type == 'variable' and right.node_type == 'variable' and 
                left.name == right.name):
                two_node = ConstantNode(2.0)
                var_node = VariableNode(left.name)
                return OperationNode('MUL', var_node, two_node)
        
        # (a*x+b) + (c*x+d) → 通过同类项合并处理（已在_collect_and_merge_like_terms中）
        
        return None
    
    def _normalize_commutative(self):
        """交换律规范化：确保常数在右边"""
        if self.op not in self.COMMUTATIVE_OPS:
            return
        
        # 交换律：常数在右边，变量在左边
        if self.left.node_type == 'constant' and self.right.node_type != 'constant':
            # 交换左右
            self.left, self.right = self.right, self.left
            self.left.parent = self
            self.right.parent = self
        
        # 多层规范化：确保操作符顺序一致
        if self.left.node_type == 'operation' and self.left.op == self.op:
            left_left = self.left.left
            left_right = self.left.right
            
            # 重组为更标准的形式
            if left_left.node_type == 'variable' and self.right.node_type == 'constant':
                # (x + a) + b => x + (a+b)
                new_right = OperationNode(self.op, left_right, self.right)
                new_right = new_right.simplify()
                
                self.left = left_left
                self.right = new_right
                self.left.parent = self
                self.right.parent = self
    
    def to_expression(self):
        """转换为可读表达式 - Evolution Core 3.2增强
        
        修复：避免递归调用to_expression导致的无限递归
        """
        if self.left is None or self.right is None:
            return "0"
        
        # 使用简化后的节点
        simplified = self.simplify()
        if simplified is not self:
            # 如果简化为叶节点，直接调用叶节点的to_expression
            if simplified.node_type != 'operation':
                return simplified.to_expression()
            
            # 操作节点：直接构建表达式
            left = simplified.left
            right = simplified.right
            op = simplified.op
            
            left_expr = left.to_expression() if left else "0"
            right_expr = right.to_expression() if right else "0"
            
            if op == 'ADD':
                return f"{left_expr} + {right_expr}"
            elif op == 'SUB':
                if right_expr.startswith('-'):
                    return f"{left_expr} + ({right_expr})"
                return f"{left_expr} - {right_expr}"
            elif op == 'MUL':
                if right.node_type == 'constant':
                    return f"{right_expr} * {left_expr}"
                return f"{left_expr} * {right_expr}"
            elif op == 'DIV':
                return f"{left_expr} / {right_expr}"
            
            return f"{op}({left_expr}, {right_expr})"
        
        left_expr = self.left.to_expression()
        right_expr = self.right.to_expression()
        
        # 规范化输出格式
        if self.op == 'ADD':
            return f"{left_expr} + {right_expr}"
        elif self.op == 'SUB':
            if right_expr.startswith('-'):
                return f"{left_expr} + ({right_expr})"
            return f"{left_expr} - {right_expr}"
        elif self.op == 'MUL':
            if self.right.node_type == 'constant':
                return f"{right_expr} * {left_expr}"
            return f"{left_expr} * {right_expr}"
        elif self.op == 'DIV':
            return f"{left_expr} / {right_expr}"
        
        return f"{self.op}({left_expr}, {right_expr})"
    
    def to_vm_code(self):
        """转换为VM代码（表达式求值后留在栈顶）"""
        if self.left is None or self.right is None:
            return "PUSH 0"
        
        code_parts = []
        
        # 左子树代码（结果留在栈顶）
        left_code = self.left.to_vm_code()
        code_parts.append(left_code)
        
        # 右子树代码（结果留在栈顶）
        right_code = self.right.to_vm_code()
        code_parts.append(right_code)
        
        # 操作（弹出两个操作数，压入结果）
        if self.op == 'ADD':
            code_parts.append("ADD")
        elif self.op == 'SUB':
            code_parts.append("SUB")
        elif self.op == 'MUL':
            code_parts.append("MUL")
        elif self.op == 'DIV':
            code_parts.append("DIV")
        
        return '\n'.join(code_parts)
    
    def copy(self):
        new_left = self.left.copy() if self.left else None
        new_right = self.right.copy() if self.right else None
        
        new_node = OperationNode(self.op, new_left, new_right)
        new_node.protected = self.protected
        new_node.fitness_contribution = self.fitness_contribution
        
        return new_node
    
    def get_depth(self):
        left_depth = self.left.get_depth() if self.left else 0
        right_depth = self.right.get_depth() if self.right else 0
        return max(left_depth, right_depth) + 1
    
    def get_size(self):
        left_size = self.left.get_size() if self.left else 0
        right_size = self.right.get_size() if self.right else 0
        return left_size + right_size + 1
    
    def get_nodes(self):
        nodes = [self]
        if self.left:
            nodes.extend(self.left.get_nodes())
        if self.right:
            nodes.extend(self.right.get_nodes())
        return nodes
    
    def __repr__(self):
        return f"Op({self.op}, {self.left}, {self.right})"


class TreeGenome:
    """Tree Genome - 结构化程序进化
    
    Evolution Core 3.0的核心实现
    """
    
    def __init__(self, root=None):
        self.root = root
        self.fitness = 0.0
        self.mdl_score = 0.0  # MDL复杂度评分
        self.protected_nodes = []  # Protected Gene列表
        self.archive_key = None  # Archive Memory索引
    
    @staticmethod
    def create_random(max_depth=3):
        """创建随机Tree Genome - Evolution Core 3.2增强
        
        智能随机生成：偏向合理数学结构
        """
        root = TreeGenome._generate_random_tree(max_depth)
        return TreeGenome(root)
    
    @staticmethod
    def _generate_random_tree(depth, is_left=True):
        """生成随机树结构 - Evolution Core 3.2增强
        
        智能随机生成：
        - 增加ADD/MUL概率，减少DIV
        - 右子树更浅（模拟 a*x+b 形式）
        - 30%概率提前终止（避免过深）
        - 常数偏向简单值（整数）
        """
        # 30%概率提前终止（避免过深）
        if depth == 0 or random.random() < 0.3:
            # 叶节点：40%变量，60%常数
            if random.random() < 0.4:
                return VariableNode('x')
            else:
                # 常数偏向简单值
                if random.random() < 0.7:
                    value = float(random.randint(-5, 5))
                else:
                    value = random.uniform(-5, 5)
                return ConstantNode(value)
        
        # 操作节点：偏向ADD和MUL（数学上更有用）
        op_weights = [0.35, 0.25, 0.30, 0.10]  # ADD, SUB, MUL, DIV
        ops = ['ADD', 'SUB', 'MUL', 'DIV']
        op = random.choices(ops, weights=op_weights, k=1)[0]
        
        # 不平衡树：左子树更深，右子树更浅（模拟 a*x+b 形式）
        if is_left:
            left_depth = depth - 1
            right_depth = max(depth - 2, 0)
        else:
            left_depth = max(depth - 2, 0)
            right_depth = max(depth - 2, 0)
        
        left = TreeGenome._generate_random_tree(left_depth, is_left=True)
        right = TreeGenome._generate_random_tree(right_depth, is_left=False)
        
        return OperationNode(op, left, right)
    
    def evaluate(self, inputs):
        """评估Tree Genome在多个输入上的表现"""
        total_error = 0.0
        valid_count = 0
        
        for x in inputs:
            try:
                result = self.root.evaluate(x)
                total_error += result ** 2  # 基础误差（未指定target）
                valid_count += 1
            except Exception:
                pass
        
        return {
            'valid_count': valid_count,
            'avg_error': total_error / max(valid_count, 1)
        }
    
    def evaluate_with_target(self, inputs, target_fn):
        """带目标函数的评估"""
        total_error = 0.0
        valid_count = 0
        
        for x in inputs:
            try:
                actual = self.root.evaluate(x)
                target = target_fn(x)
                error = (actual - target) ** 2
                total_error += error
                valid_count += 1
            except Exception:
                pass
        
        mse = total_error / max(valid_count, 1)
        accuracy = 1.0 / (1.0 + mse)
        
        return {
            'mse': mse,
            'accuracy': accuracy,
            'valid_count': valid_count
        }
    
    def calculate_fitness(self, inputs, target_fn, mdl_weight=0.1):
        """计算Fitness + MDL复杂度 + Human Readability
        
        Evolution Core 3.1: 三维度评分
        - accuracy
        - simplicity (MDL)
        - readability (human readability score)
        """
        eval_result = self.evaluate_with_target(inputs, target_fn)
        
        if eval_result['valid_count'] == 0:
            self.fitness = 0.0
            self.mdl_score = float('inf')
            return self.fitness
        
        # 简化表达式
        simplified = self.simplify()
        
        # MDL复杂度评分（归一化）
        complexity = self.get_complexity()
        normalized_complexity = complexity / 20.0
        mdl_penalty = normalized_complexity * mdl_weight
        
        # Human Readability Score
        readability_score = self._calculate_readability_score()
        readability_bonus = readability_score * 0.1
        
        # Fitness = accuracy - MDL_penalty + readability_bonus
        self.fitness = eval_result['accuracy'] - mdl_penalty + readability_bonus
        self.mdl_score = eval_result['mse'] + complexity * 0.05
        
        return max(self.fitness, 0.001)
    
    def simplify(self):
        """符号简化"""
        if self.root is None:
            return self
        
        simplified_root = self.root.simplify() if hasattr(self.root, 'simplify') else self.root
        
        # 创建简化后的新Genome
        new_genome = TreeGenome(simplified_root)
        new_genome.fitness = self.fitness
        new_genome.mdl_score = self.mdl_score
        new_genome.protected_nodes = self.protected_nodes
        
        return new_genome
    
    def _calculate_readability_score(self):
        """Human Readability Score
        
        奖励：
        - 短表达式
        - 少常数
        - 标准形式（如 2*x+1）
        """
        expr = self.to_expression()
        
        # 1. 表达式长度评分（越短越好）
        length_score = 1.0 - min(len(expr) / 30.0, 1.0)
        
        # 2. 常数数量评分（越少越好）
        constant_count = self._count_constants()
        constant_score = 1.0 - min(constant_count / 5.0, 1.0)
        
        # 3. 标准形式评分（检查是否符合标准形式）
        pattern_score = self._check_standard_pattern(expr)
        
        # 综合评分
        readability = (length_score * 0.3 + constant_score * 0.3 + pattern_score * 0.4)
        
        return readability
    
    def _count_constants(self):
        """计算常数节点数量"""
        if self.root is None:
            return 0
        
        nodes = self.root.get_nodes()
        return sum(1 for n in nodes if n.node_type == 'constant')
    
    def _check_standard_pattern(self, expr):
        """检查标准形式模式
        
        标准形式示例：
        - 2*x+1
        - 3*x-5
        - x+7
        - 2*x
        """
        # 标准线性形式: a*x+b
        if expr.replace(' ', '') in ['2*x+1', '3*x+1', 'x+1', 'x-1', '2*x', '3*x']:
            return 1.0
        
        # 检查是否接近标准形式
        import re
        
        # 检查模式: 数字*x+数字 或 数字*x-数字
        patterns = [
            r'^\d+\*x[\+\-]\d+$',  # 如 2*x+1
            r'^x[\+\-]\d+$',       # 如 x+1
            r'^\d+\*x$',           # 如 2*x
        ]
        
        for pattern in patterns:
            if re.match(pattern, expr.replace(' ', '')):
                return 0.9
        
        return 0.5
    
    def get_complexity(self):
        """计算程序复杂度"""
        if self.root is None:
            return 0
        
        size = self.root.get_size()
        depth = self.root.get_depth()
        
        # MDL复杂度 = size + depth * 2
        return size + depth * 2
    
    def to_expression(self):
        """转换为可读表达式"""
        if self.root is None:
            return "0"
        return self.root.to_expression()
    
    def to_vm_code(self):
        """转换为VM代码"""
        if self.root is None:
            return "PUSH 0"
        return self.root.to_vm_code()
    
    def copy(self):
        """深拷贝Tree Genome"""
        new_root = self.root.copy() if self.root else None
        new_genome = TreeGenome(new_root)
        new_genome.fitness = self.fitness
        new_genome.mdl_score = self.mdl_score
        new_genome.protected_nodes = [n.copy() for n in self.protected_nodes]
        new_genome.archive_key = self.archive_key
        return new_genome
    
    # ========================================================================
    # 结构Mutation - Evolution Core 3.0
    # ========================================================================
    
    def mutate(self, mutation_rate=0.3, fitness=0.0):
        """结构化变异
        
        包括：
        - replace_node: 替换节点
        - add_branch: 添加分支
        - remove_branch: 删除分支
        - optimize_constant: 优化常数
        """
        if self.root is None:
            return
        
        # Protected Gene机制：高fitness降低变异率
        effective_rate = mutation_rate * (1 - fitness * 0.5)
        
        nodes = self.root.get_nodes()
        
        for node in nodes:
            if random.random() < effective_rate:
                # Protected Gene检查
                if node.protected:
                    if random.random() > 0.2:  # 80%概率跳过保护节点
                        continue
                
                # 选择变异类型
                mutation_type = random.random()
                
                if mutation_type < 0.3:
                    self._replace_node(node)
                elif mutation_type < 0.5:
                    self._optimize_constant(node)
                elif mutation_type < 0.7:
                    self._add_branch(node)
                else:
                    self._remove_branch(node)
    
    def _replace_node(self, node):
        """替换节点"""
        if node.node_type == 'constant':
            # 常数微调
            node.value += random.gauss(0, 0.5)
        elif node.node_type == 'variable':
            # 变量节点不变
            pass
        elif node.node_type == 'operation':
            # 替换操作
            new_op = random.choice(['ADD', 'SUB', 'MUL', 'DIV'])
            node.op = new_op
    
    def _optimize_constant(self, node):
        """优化常数节点"""
        if node.node_type == 'constant':
            # 连续微调
            mutation_type = random.random()
            if mutation_type < 0.6:
                node.value += random.gauss(0, 0.2)
            elif mutation_type < 0.8:
                node.value += random.gauss(0, 1.0)
            else:
                node.value = random.uniform(-5, 5)
    
    def _add_branch(self, node):
        """添加分支"""
        if node.node_type == 'constant' or node.node_type == 'variable':
            # 将叶节点扩展为操作节点
            new_op = random.choice(['ADD', 'MUL'])
            new_left = node.copy()
            new_right = TreeGenome._generate_random_tree(1)
            
            new_node = OperationNode(new_op, new_left, new_right)
            
            # 替换原节点
            if node.parent:
                if node.parent.left == node:
                    node.parent.left = new_node
                else:
                    node.parent.right = new_node
                new_node.parent = node.parent
            else:
                self.root = new_node
    
    def _remove_branch(self, node):
        """删除分支"""
        if node.node_type == 'operation':
            # 将操作节点简化为叶节点
            if random.random() < 0.5 and node.left:
                new_node = node.left.copy()
            elif node.right:
                new_node = node.right.copy()
            else:
                return
            
            # 替换原节点
            if node.parent:
                if node.parent.left == node:
                    node.parent.left = new_node
                else:
                    node.parent.right = new_node
                new_node.parent = node.parent
            else:
                self.root = new_node
    
    def protect_high_fitness_nodes(self, threshold=0.8):
        """保护高fitness贡献节点"""
        if self.root is None:
            return
        
        nodes = self.root.get_nodes()
        
        for node in nodes:
            if node.fitness_contribution >= threshold:
                node.protected = True
                self.protected_nodes.append(node)
    
    def __repr__(self):
        return f"TreeGenome(expr={self.to_expression()}, fitness={self.fitness:.4f}, mdl={self.mdl_score:.2f})"
    
    def __len__(self):
        return self.get_complexity()


# ============================================================================
# Archive Memory - Evolution Core 3.0
# ============================================================================

class ArchiveMemory:
    """Archive Memory - 保存历史优秀程序结构
    
    Evolution Core 3.1: 升级保存抽象程序模式
    """
    
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.archive = {}  # key: expression, value: genome
        self.pattern_archive = {}  # key: abstract pattern, value: list of genomes
        self.fitness_history = []
    
    def add(self, genome):
        """添加优秀程序到Archive - Evolution Core 3.1增强"""
        if genome.fitness < 0.1:  # 只保存高fitness程序
            return
        
        # 先简化
        simplified = genome.simplify()
        expression = simplified.to_expression()
        
        # 提取抽象模式
        pattern = self._extract_abstract_pattern(expression)
        
        # 检查是否已存在
        if expression in self.archive:
            # 更新fitness
            if genome.fitness > self.archive[expression].fitness:
                self.archive[expression] = genome.copy()
            return
        
        # 添加新程序
        if len(self.archive) >= self.max_size:
            # 移除最低fitness的程序
            worst_key = min(self.archive.keys(), 
                          key=lambda k: self.archive[k].fitness)
            del self.archive[worst_key]
        
        genome.archive_key = expression
        self.archive[expression] = genome.copy()
        
        # 添加到模式Archive
        if pattern not in self.pattern_archive:
            self.pattern_archive[pattern] = []
        
        self.pattern_archive[pattern].append(genome.copy())
        
        # 限制每个模式最多保存5个实例
        if len(self.pattern_archive[pattern]) > 5:
            # 移除最低fitness的实例
            self.pattern_archive[pattern].sort(key=lambda g: g.fitness, reverse=True)
            self.pattern_archive[pattern] = self.pattern_archive[pattern][:5]
        
        # 记录fitness历史
        self.fitness_history.append({
            'expression': expression,
            'pattern': pattern,
            'fitness': genome.fitness,
            'mdl': genome.mdl_score
        })
    
    def _extract_abstract_pattern(self, expression):
        """提取抽象模式
        
        示例：
        - 2*x+1 -> a*x+b
        - x+3 -> x+b
        - 3*x -> a*x
        """
        import re
        
        # 替换具体数字为抽象符号
        # 数字*x -> a*x
        pattern = re.sub(r'\d+\*', 'a*', expression)
        
        # *数字 -> *b
        pattern = re.sub(r'\*\d+', '*b', pattern)
        
        # +数字 -> +b
        pattern = re.sub(r'\+\d+', '+b', pattern)
        
        # -数字 -> -b
        pattern = re.sub(r'\-\d+', '-b', pattern)
        
        return pattern
    
    def get_best(self):
        """获取最佳程序"""
        if not self.archive:
            return None
        
        best_key = max(self.archive.keys(), 
                      key=lambda k: self.archive[k].fitness)
        return self.archive[best_key].copy()
    
    def get_pattern_instances(self, pattern):
        """获取特定模式的实例"""
        if pattern not in self.pattern_archive:
            return []
        
        return [g.copy() for g in self.pattern_archive[pattern]]
    
    def get_similar(self, genome, threshold=0.8):
        """获取相似程序（用于 crossover）"""
        similar_genomes = []
        
        for key, archived_genome in self.archive.items():
            # 计算结构相似度
            similarity = self._calculate_similarity(genome, archived_genome)
            if similarity >= threshold:
                similar_genomes.append(archived_genome.copy())
        
        return similar_genomes
    
    def _calculate_similarity(self, genome1, genome2):
        """计算程序结构相似度"""
        expr1 = genome1.simplify().to_expression()
        expr2 = genome2.simplify().to_expression()
        
        # 简单字符串相似度
        common_chars = sum(1 for c in expr1 if c in expr2)
        max_len = max(len(expr1), len(expr2))
        
        return common_chars / max_len if max_len > 0 else 0.0
    
    def get_statistics(self):
        """获取Archive统计"""
        if not self.archive:
            return {'size': 0, 'avg_fitness': 0, 'avg_mdl': 0, 'patterns': 0}
        
        fitnesses = [g.fitness for g in self.archive.values()]
        mdls = [g.mdl_score for g in self.archive.values()]
        
        return {
            'size': len(self.archive),
            'patterns': len(self.pattern_archive),
            'avg_fitness': np.mean(fitnesses),
            'avg_mdl': np.mean(mdls),
            'best_fitness': max(fitnesses),
            'worst_fitness': min(fitnesses),
            'pattern_list': list(self.pattern_archive.keys())
        }

_OP_CODES = {
    'PUSH': 1, 'ADD': 2, 'SUB': 3, 'MUL': 4, 'DIV': 5,
    'SHAPE': 6, 'POP': 7, 'STORE': 8, 'LOAD': 9, 'PRINT': 10,
    'TAPE': 11, 'UNTAPE': 12, 'GRAD': 13, 'HALT': 14, 'JMP': 15,
    'JMP_IF': 16, 'RELU': 17, 'NEG': 18, 'POW': 19, 'MATMUL': 20
}

_CODE_OPS = {v: k for k, v in _OP_CODES.items()}

_VAR_PREFIX = 100


class Genome:
    def __init__(self, genes=None, length=None):
        if genes is not None:
            self.genes = np.array(genes, dtype=np.float32)
        elif length is not None:
            self.genes = self._random_genes(length)
        else:
            self.genes = np.array([], dtype=np.float32)
        self.fitness = None

    @staticmethod
    def _random_genes(length=None):
        """生成有意义的种子指令序列，而不是随机填充
        
        生成策略：
        - 长度灵活（默认5-20条有效指令，对应基因数约8-30）
        - 确保操作码和操作数配对正确
        - 偏向有用的操作组合（LOAD/PUSH/ADD/MUL/STORE）
        
        参数：
            length: 如果指定，生成大约 length 个基因（向后兼容）
                   如果不指定，随机生成 5-20 条指令
        """
        if length is not None:
            # 指定长度：尽量接近目标基因数，同时保持指令结构合理
            genes = []
            useful_ops = ['LOAD', 'PUSH', 'ADD', 'SUB', 'MUL', 'STORE', 'POP']
            op_weights = [0.2, 0.25, 0.15, 0.1, 0.15, 0.1, 0.05]
            
            target = max(3, length)
            while len(genes) < target:
                op = random.choices(useful_ops, weights=op_weights, k=1)[0]
                genes.append(float(_OP_CODES[op]))
                
                if op == 'PUSH':
                    val = random.uniform(-5, 5)
                    if random.random() < 0.4:
                        val = float(random.randint(-5, 5))
                    genes.append(val)
                elif op in ('LOAD', 'STORE'):
                    var_idx = random.randint(0, 3)
                    genes.append(float(_VAR_PREFIX + var_idx))
            
            # 截断到目标长度
            return np.array(genes[:length], dtype=np.float32)
        
        # 不指定长度：灵活长度的指令序列
        num_ops = random.randint(5, 20)
        genes = []
        useful_ops = ['LOAD', 'PUSH', 'ADD', 'SUB', 'MUL', 'STORE', 'POP']
        op_weights = [0.2, 0.25, 0.15, 0.1, 0.15, 0.1, 0.05]
        
        for _ in range(num_ops):
            op = random.choices(useful_ops, weights=op_weights, k=1)[0]
            genes.append(float(_OP_CODES[op]))
            
            if op == 'PUSH':
                val = random.uniform(-5, 5)
                if random.random() < 0.4:
                    val = float(random.randint(-5, 5))
                genes.append(val)
            elif op in ('LOAD', 'STORE'):
                var_idx = random.randint(0, 3)
                genes.append(float(_VAR_PREFIX + var_idx))
        
        return np.array(genes, dtype=np.float32)

    def decode(self):
        script = []
        i = 0
        while i < len(self.genes):
            gene = self.genes[i]
            if gene.is_integer() and int(gene) in _CODE_OPS:
                op = _CODE_OPS[int(gene)]
                if op == 'PUSH':
                    if i + 1 < len(self.genes):
                        next_gene = self.genes[i + 1]
                        if not (next_gene.is_integer() and int(next_gene) in _CODE_OPS):
                            script.append(f"PUSH {next_gene}")
                            i += 2
                            continue
                    script.append("PUSH 0")
                    i += 1
                elif op in ('STORE', 'LOAD', 'JMP', 'JMP_IF'):
                    if i + 1 < len(self.genes):
                        var_gene = self.genes[i + 1]
                        if var_gene.is_integer() and int(var_gene) >= _VAR_PREFIX:
                            var_idx = int(var_gene) - _VAR_PREFIX
                            var_name = chr(ord('a') + var_idx)
                            script.append(f"{op} {var_name}")
                            i += 2
                            continue
                    script.append(f"{op} a")
                    i += 1
                elif op == 'SHAPE':
                    if i + 2 < len(self.genes):
                        dim1 = self.genes[i + 1]
                        dim2 = self.genes[i + 2]
                        if dim1.is_integer() and dim2.is_integer():
                            script.append(f"SHAPE ({int(dim1)},{int(dim2)})")
                            i += 3
                            continue
                    script.append("SHAPE (1,1)")
                    i += 1
                else:
                    script.append(op)
                    i += 1
            else:
                i += 1
        return '\n'.join(script)

    def execute(self, input_vars=None):
        vm = NeoGlyphVM()
        if input_vars:
            pre_script = []
            for name, value in input_vars.items():
                if isinstance(value, (list, np.ndarray)):
                    pre_script.append(f"PUSH {' '.join(map(str, value))}")
                else:
                    pre_script.append(f"PUSH {value}")
                pre_script.append(f"STORE {name}")
            full_script = '\n'.join(pre_script) + '\n' + self.decode()
        else:
            full_script = self.decode()
        vm.run(full_script)
        return vm

    def evaluate(self, target_fn, input_vars=None):
        try:
            vm = self.execute(input_vars)
            result = vm.vars.get('out')
            if result is None and vm.vars:
                result = list(vm.vars.values())[-1]
            if result is None and vm.stack:
                result = vm.stack[-1]
            if result is not None:
                target = target_fn(vm)
                if isinstance(target, np.ndarray):
                    target = target.astype(np.float32)
                else:
                    target = np.array([target], dtype=np.float32)
                error = np.mean((result.data - target) ** 2)
                self.fitness = 1.0 / (1.0 + error)
            else:
                self.fitness = 0.0
        except Exception:
            self.fitness = 0.0
        return self.fitness

    @staticmethod
    def crossover(parent1, parent2, method='single'):
        if len(parent1.genes) == 0 or len(parent2.genes) == 0:
            return Genome(parent1.genes.copy())

        if method == 'single':
            point = random.randint(1, min(len(parent1.genes), len(parent2.genes)) - 1)
            child_genes = np.concatenate([parent1.genes[:point], parent2.genes[point:]])
        elif method == 'uniform':
            child_genes = np.array([
                parent1.genes[i] if random.random() < 0.5 else parent2.genes[i]
                for i in range(min(len(parent1.genes), len(parent2.genes)))
            ])
        else:
            child_genes = parent1.genes.copy()

        return Genome(child_genes)

    def mutate(self, mutation_rate=0.1, mutation_std=0.5, max_length=100, protect_important=True, fitness=0.0):
        elite_factor = max(0, min(1, fitness * 3))
        
        i = 0
        while i < len(self.genes):
            effective_rate = mutation_rate * (1 - elite_factor * 0.5)
            
            if random.random() < effective_rate:
                gene_type = int(self.genes[i])
                
                if gene_type in _CODE_OPS:
                    op = _CODE_OPS[gene_type]
                    
                    if protect_important and op in ['LOAD', 'STORE', 'ADD', 'MUL', 'PUSH']:
                        elite_protect = random.random() > (0.15 + elite_factor * 0.7)
                        if elite_protect:
                            i += 1
                            continue
                    
                    mutation_type = random.random()
                    if mutation_type < 0.7:
                        self.genes[i] = random.randint(1, 20)
                    else:
                        if elite_factor > 0.5:
                            new_op = random.choice(['LOAD', 'STORE', 'ADD', 'MUL', 'PUSH', 'POP'])
                        else:
                            new_op = random.choice(['LOAD', 'STORE', 'ADD', 'MUL', 'PUSH', 'POP', 'HALT'])
                        self.genes[i] = _OP_CODES[new_op]
                    i += 1
                
                elif gene_type >= _VAR_PREFIX:
                    if protect_important:
                        if random.random() > 0.3:
                            i += 1
                            continue
                    self.genes[i] = random.randint(_VAR_PREFIX, _VAR_PREFIX + 10)
                    i += 1
                
                else:
                    mutation_type = random.random()
                    if mutation_type < 0.5:
                        self.genes[i] += random.gauss(0, mutation_std * 0.15)
                    elif mutation_type < 0.75:
                        self.genes[i] += random.gauss(0, mutation_std * 0.5)
                    elif mutation_type < 0.9:
                        self.genes[i] += random.gauss(0, mutation_std)
                    elif mutation_type < 0.97:
                        self.genes[i] = random.uniform(-3, 3)
                    else:
                        val = random.uniform(-10, 10)
                        while val.is_integer():
                            val = random.uniform(-10, 10)
                        self.genes[i] = val
                    i += 1
            
            elif protect_important and i + 1 < len(self.genes):
                gene_type = int(self.genes[i])
                if gene_type in _CODE_OPS and _CODE_OPS[gene_type] == 'PUSH':
                    if random.random() < 0.1:
                        self.genes[i+1] += random.gauss(0, mutation_std * 0.15)
            i += 1
    
    def _generate_random_gene(self):
        rand = random.random()
        if rand < 0.25:
            return float(random.randint(1, 20))
        elif rand < 0.40:
            return float(random.randint(_VAR_PREFIX, _VAR_PREFIX + 10))
        else:
            val = random.uniform(-10, 10)
            while val.is_integer():
                val = random.uniform(-10, 10)
            return val

    @staticmethod
    def selection(population, method='tournament', tournament_size=3):
        if method == 'tournament':
            candidates = random.sample(population, tournament_size)
            return max(candidates, key=lambda g: g.fitness)
        elif method == 'roulette':
            total_fitness = sum(g.fitness for g in population)
            if total_fitness == 0:
                return random.choice(population)
            r = random.random() * total_fitness
            for g in population:
                r -= g.fitness
                if r <= 0:
                    return g
            return population[-1]
        else:
            return random.choice(population)

    def __repr__(self):
        return f"Genome(length={len(self.genes)}, fitness={self.fitness:.4f})"

    def __len__(self):
        return len(self.genes)


class GeneticOptimizer:
    def __init__(self, pop_size=50, gene_length=20, mutation_rate=0.1):
        self.pop_size = pop_size
        self.gene_length = gene_length
        self.mutation_rate = mutation_rate
        self.population = [Genome(length=gene_length) for _ in range(pop_size)]
        self.generation = 0

    def evolve(self, target_fn, input_vars=None, generations=100, verbose=False):
        for gen in range(generations):
            self.generation = gen + 1

            for genome in self.population:
                genome.evaluate(target_fn, input_vars)

            valid = [g for g in self.population if g.fitness is not None]
            if valid:
                best = max(valid, key=lambda g: g.fitness)
            else:
                best = self.population[0]

            if verbose and gen % 10 == 0:
                valid_fitness = [g.fitness for g in self.population if g.fitness is not None]
                avg_fitness = sum(valid_fitness) / len(valid_fitness) if valid_fitness else 0
                print(f"Generation {gen}: Best={best.fitness:.4f}, Avg={avg_fitness:.4f}")

            new_population = [best]

            while len(new_population) < self.pop_size:
                parent1 = Genome.selection(self.population)
                parent2 = Genome.selection(self.population)
                child = Genome.crossover(parent1, parent2)
                child.mutate(self.mutation_rate)
                new_population.append(child)

            self.population = new_population

        valid = [g for g in self.population if g.fitness is not None]
        return max(valid, key=lambda g: g.fitness) if valid else self.population[0]

    def get_best(self):
        return max(self.population, key=lambda g: g.fitness)
