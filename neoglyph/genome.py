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
    
    def to_vm_code(self, var_map=None):
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
    
    def to_vm_code(self, var_map=None):
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
    """变量节点 - 支持单变量和多变量
    
    单变量: evaluate(3.0) → 3.0
    多变量: evaluate({'x': 3.0, 'y': 4.0}) → 按 name 查找
    """
    
    def __init__(self, name='x'):
        super().__init__()
        self.name = name
        self.node_type = 'variable'
    
    def evaluate(self, x):
        if isinstance(x, dict):
            return float(x.get(self.name, 0.0))
        return np.asarray(x, dtype=np.float64)
    
    def to_expression(self):
        return self.name
    
    def to_vm_code(self, var_map=None):
        if var_map is None:
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
    
    支持自定义算子注册。
    
    新增：
    - 同类项合并
    - 多级表达式化简
    - 简化结果缓存
    - 自定义算子接口
    """
    
    OPERATIONS = {
        'ADD': lambda a, b: a + b,
        'SUB': lambda a, b: a - b,
        'MUL': lambda a, b: a * b,
        'DIV': lambda a, b: (
            np.divide(a, b, out=np.zeros_like(np.asarray(a, dtype=np.float64)), 
                      where=np.abs(np.asarray(b, dtype=np.float64)) > 1e-15)),
        'SIN': lambda a: np.sin(a),
        'COS': lambda a: np.cos(a),
        'EXP': lambda a: np.exp(np.clip(a, -80, 80)),
        'LOG': lambda a: np.log(np.maximum(a, 1e-10)),
        'NEG': lambda a: -a,
    }
    
    UNARY_OPS = ['SIN', 'COS', 'EXP', 'LOG', 'NEG']
    COMMUTATIVE_OPS = ['ADD', 'MUL']
    
    @staticmethod
    def register_operator(name, fn, is_unary=False, commutative=False):
        """注册自定义算子
        
        Parameters
        ----------
        name : str
            算子名称（如 'ABS', 'SQRT'）
        fn : callable
            算子函数。二元算子接受 (a, b)，一元算子接受 (a)
        is_unary : bool
            是否为一元算子
        commutative : bool
            是否为可交换算子
        
        Examples
        --------
        >>> OperationNode.register_operator('ABS', lambda a: np.abs(a), is_unary=True)
        >>> OperationNode.register_operator('SQUARE', lambda a: a**2, is_unary=True)
        >>> OperationNode.register_operator('MAX', lambda a, b: np.maximum(a, b))
        """
        OperationNode.OPERATIONS[name] = fn
        if is_unary:
            if name not in OperationNode.UNARY_OPS:
                OperationNode.UNARY_OPS.append(name)
        if commutative:
            if name not in OperationNode.COMMUTATIVE_OPS:
                OperationNode.COMMUTATIVE_OPS.append(name)
    
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
        if self.op in self.UNARY_OPS:
            if self.left is None:
                return 0.0
            val = self.left.evaluate(x)
            try:
                return self.OPERATIONS[self.op](val)
            except Exception:
                return 0.0
        
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
            if self.op in self.UNARY_OPS:
                # 一元操作：只需要左子节点
                if self.left is None:
                    self._simplified_cache = self
                    self._cache_dirty = False
                    return self
                # 简化子节点（仅操作为节点）
                if self.left.node_type == 'operation':
                    simplified_left = self.left.simplify()
                    if simplified_left != self.left:
                        self.left = simplified_left
                        self.left.parent = self
                
                # 一元操作简化规则
                if self.op == 'NEG' and self.left.node_type == 'constant':
                    result = ConstantNode(-self.left.value)
                    self._simplified_cache = result
                    self._cache_dirty = False
                    return result
                if self.op == 'EXP' and self.left.node_type == 'constant' and self.left.value == 0.0:
                    self._simplified_cache = ConstantNode(1.0)
                    self._cache_dirty = False
                    return ConstantNode(1.0)
                if self.op == 'LOG' and self.left.node_type == 'constant' and self.left.value == 1.0:
                    self._simplified_cache = ConstantNode(0.0)
                    self._cache_dirty = False
                    return ConstantNode(0.0)
                if self.op == 'SIN' and self.left.node_type == 'constant' and self.left.value == 0.0:
                    self._simplified_cache = ConstantNode(0.0)
                    self._cache_dirty = False
                    return ConstantNode(0.0)
                
                self._simplified_cache = self
                self._cache_dirty = False
                return self
            
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
        left_x_coeff, left_const, left_var = left_info
        right_x_coeff, right_const, right_var = right_info
        
        # 变量名不同则不合并（如 x + y 不应合并为 2*x）
        if left_var != right_var:
            return None
        
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
                x_part = VariableNode(left_var)
            else:
                x_part = OperationNode('MUL', 
                                        VariableNode(left_var), 
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
        """提取节点的项信息 (x系数, 常数项, 变量名)
        
        返回 (x_coefficient, constant_term, var_name) 或 None
        var_name 用于区分不同变量，防止 x+y 被错误合并为 2*x
        """
        if node is None:
            return None
        
        if node.node_type == 'constant':
            return (0.0, node.value, None)
        
        if node.node_type == 'variable':
            return (1.0, 0.0, node.name)
        
        if node.node_type == 'operation':
            # 一元操作（如SIN/COS/EXP/LOG/NEG）不作为项处理
            if node.op in OperationNode.UNARY_OPS:
                return None
            
            if node.op == 'MUL':
                # 确保子节点非空
                if node.left is None or node.right is None:
                    return None
                # a*x 或 x*a
                if node.left.node_type == 'constant' and node.right.node_type == 'variable':
                    return (node.left.value, 0.0, node.right.name)
                if node.left.node_type == 'variable' and node.right.node_type == 'constant':
                    return (node.right.value, 0.0, node.left.name)
                # 两个都是常数
                if node.left.node_type == 'constant' and node.right.node_type == 'constant':
                    return (0.0, node.left.value * node.right.value, None)
            
            elif node.op == 'ADD':
                left_info = self._extract_term_info(node.left)
                right_info = self._extract_term_info(node.right)
                if left_info and right_info:
                    # 只有变量名相同时才合并系数
                    if left_info[2] == right_info[2]:
                        return (left_info[0] + right_info[0], left_info[1] + right_info[1], left_info[2])
            
            elif node.op == 'SUB':
                left_info = self._extract_term_info(node.left)
                right_info = self._extract_term_info(node.right)
                if left_info and right_info:
                    if left_info[2] == right_info[2]:
                        return (left_info[0] - right_info[0], left_info[1] - right_info[1], left_info[2])
        
        return None
    
    def _canonicalize_constants(self):
        """常数规范化：近似整数恢复成整数"""
        if self.left is not None and self.left.node_type == 'constant':
            self.left.value = self._canonicalize_value(self.left.value)
        if self.right is not None and self.right.node_type == 'constant':
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
        if self.left is None or self.right is None:
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
            
            if left_left is None or self.right is None:
                return
            
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
        # 一元操作
        if self.op in self.UNARY_OPS:
            if self.left is None:
                return "0"
            if self.left.node_type == 'operation':
                simplified = self.left.simplify()
                inner = simplified.to_expression()
            else:
                inner = self.left.to_expression()
            if self.op == 'NEG':
                return f"-({inner})" if inner.startswith('-') else f"-{inner}"
            return f"{self.op.lower()}({inner})"
        
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
    
    def to_vm_code(self, var_map=None):
        """转换为VM代码（表达式求值后留在栈顶）"""
        # 一元操作
        if self.op in self.UNARY_OPS:
            left_code = self.left.to_vm_code(var_map) if self.left else "PUSH 0"
            return f"{left_code}\n{self.op}"
        
        if self.left is None or self.right is None:
            return "PUSH 0"
        
        code_parts = []
        
        # 左子树代码（结果留在栈顶）
        left_code = self.left.to_vm_code(var_map)
        code_parts.append(left_code)
        
        # 右子树代码（结果留在栈顶）
        right_code = self.right.to_vm_code(var_map)
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
    def create_random(max_depth=3, variable_names=None):
        """创建随机Tree Genome - Evolution Core 3.2增强
        
        智能随机生成：偏向合理数学结构
        
        variable_names: 变量名列表，如 ['x', 'y']，默认 ['x']
        """
        if variable_names is None:
            variable_names = ['x']
        root = TreeGenome._generate_random_tree(max_depth, variable_names=variable_names)
        return TreeGenome(root)
    
    @staticmethod
    def _generate_random_tree(depth, is_left=True, variable_names=None):
        """生成随机树结构 - Evolution Core 3.2增强
        
        智能随机生成：
        - 增加ADD/MUL概率，减少DIV
        - 右子树更浅（模拟 a*x+b 形式）
        - 30%概率提前终止（避免过深）
        - 常数偏向简单值（整数）
        - 多变量支持：从 variable_names 中随机选择
        """
        if variable_names is None:
            variable_names = ['x']
        
        # 30%概率提前终止（避免过深）
        if depth == 0 or random.random() < 0.3:
            # 叶节点：40%变量，60%常数
            if random.random() < 0.4:
                var_name = random.choice(variable_names)
                return VariableNode(var_name)
            else:
                # 常数偏向简单值
                if random.random() < 0.7:
                    value = float(random.randint(-5, 5))
                else:
                    value = random.uniform(-5, 5)
                return ConstantNode(value)
        
        # 操作节点：偏向ADD和MUL（数学上更有用），包含一元函数
        op_weights = [0.25, 0.15, 0.20, 0.05, 0.10, 0.10, 0.10, 0.05]  # ADD, SUB, MUL, DIV, SIN, COS, EXP, LOG
        ops = ['ADD', 'SUB', 'MUL', 'DIV', 'SIN', 'COS', 'EXP', 'LOG']
        op = random.choices(ops, weights=op_weights, k=1)[0]
        
        if op in OperationNode.UNARY_OPS:
            # 一元操作：只需要左子节点
            left = TreeGenome._generate_random_tree(depth - 1, is_left=True, variable_names=variable_names)
            return OperationNode(op, left=left)
        
        # 二元操作：不平衡树，左子树更深，右子树更浅（模拟 a*x+b 形式）
        if is_left:
            left_depth = depth - 1
            right_depth = max(depth - 2, 0)
        else:
            left_depth = max(depth - 2, 0)
            right_depth = max(depth - 2, 0)
        
        left = TreeGenome._generate_random_tree(left_depth, is_left=True, variable_names=variable_names)
        right = TreeGenome._generate_random_tree(right_depth, is_left=False, variable_names=variable_names)
        
        return OperationNode(op, left, right)
    
    def evaluate(self, inputs):
        """评估Tree Genome在多个输入上的表现
        
        支持单变量和多变量：
        - 单变量: inputs = [1.0, 2.0, 3.0]
        - 多变量: inputs = [{'x': 1.0, 'y': 2.0}, ...]
        """
        total_error = 0.0
        valid_count = 0
        
        for x in inputs:
            try:
                result = self.root.evaluate(x)
                total_error += result ** 2
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
    
    def evaluate_array(self, X, y):
        """用数组数据直接评估（用于符号回归）
        
        X: 输入数组 (n_samples,) 单变量 或 (n_samples, n_features) 多变量
        y: 目标数组 (n_samples,)
        
        返回 accuracy (1 / (1 + mse))
        
        单变量路径使用向量化计算，大幅提升性能。
        """
        if self.root is None:
            return {'mse': float('inf'), 'accuracy': 0.0, 'valid_count': 0}
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        # 收集变量名
        var_names = sorted(self._collect_variables(self.root))
        if not var_names:
            var_names = ['x']
        
        try:
            if X.ndim == 1 and len(var_names) == 1:
                # 单变量向量化路径：整数组一次性通过表达式树求值
                actual = self.root.evaluate(X)
                actual = np.asarray(actual, dtype=np.float64).ravel()
                error = (actual - y) ** 2
                mse = float(np.mean(error))
                valid_count = len(X)
            else:
                # 多变量逐样本路径
                total_error = 0.0
                valid_count = 0
                for i in range(len(X)):
                    try:
                        x_val = {var_names[j]: float(X[i, j]) for j in range(min(X.shape[1], len(var_names)))}
                        actual = self.root.evaluate(x_val)
                        error = (actual - float(y[i])) ** 2
                        total_error += error
                        valid_count += 1
                    except Exception:
                        pass
                mse = total_error / max(valid_count, 1)
        except Exception:
            return {'mse': float('inf'), 'accuracy': 0.0, 'valid_count': 0}
        
        if valid_count == 0:
            return {'mse': float('inf'), 'accuracy': 0.0, 'valid_count': 0}
        
        accuracy = 1.0 / (1.0 + mse)
        
        return {
            'mse': mse,
            'accuracy': accuracy,
            'valid_count': valid_count
        }
    
    def calculate_fitness(self, inputs, target_fn, mdl_weight=0.05):
        """计算Fitness + MDL复杂度 + Human Readability
        
        Evolution Core 3.1: 三维度评分
        - accuracy
        - simplicity (MDL)
        - readability (human readability score)
        
        改进：降低MDL惩罚权重，使进化能发现更复杂的表达式
        """
        eval_result = self.evaluate_with_target(inputs, target_fn)
        
        if eval_result['valid_count'] == 0:
            self.fitness = 0.0
            self.mdl_score = float('inf')
            return self.fitness
        
        # 简化表达式
        simplified = self.simplify()
        
        # MDL复杂度评分（归一化，降低惩罚）
        complexity = self.get_complexity()
        normalized_complexity = complexity / 30.0  # 从20放大到30，降低惩罚
        mdl_penalty = normalized_complexity * mdl_weight
        
        # Human Readability Score
        readability_score = self._calculate_readability_score()
        readability_bonus = readability_score * 0.05  # 降低可读性奖励权重
        
        # Fitness = accuracy - MDL_penalty + readability_bonus
        base_fitness = eval_result['accuracy'] - mdl_penalty + readability_bonus
        self.fitness = max(base_fitness, 0.001)
        self.mdl_score = eval_result['mse'] + complexity * 0.03  # 降低MDL惩罚
        
        return self.fitness
    
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
        """转换为VM代码，自动处理多变量
        
        收集表达式树中的所有变量名，映射到VM寄存器（a, b, c, ...），
        并在表达式求值前插入 PUSH+STORE 设置代码。
        """
        if self.root is None:
            return "PUSH 0"
        
        # 收集所有变量名
        var_names = sorted(self._collect_variables(self.root))
        var_map = {}
        for i, name in enumerate(var_names):
            var_map[name] = chr(ord('a') + i)
        
        # 表达式求值代码
        expr_code = self.root.to_vm_code(var_map)
        
        return expr_code
    
    def _collect_variables(self, node):
        """递归收集表达式树中的所有变量名"""
        names = set()
        if node is None:
            return names
        if node.node_type == 'variable':
            names.add(node.name)
        if node.node_type == 'operation':
            if node.left:
                names.update(self._collect_variables(node.left))
            if node.right:
                names.update(self._collect_variables(node.right))
        return names
    
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
        - replace_node: 替换节点（含一元操作算子）
        - add_branch: 添加分支
        - remove_branch: 删除分支
        - optimize_constant: 优化常数
        """
        if self.root is None:
            return
        
        # Protected Gene机制：高fitness降低变异率
        effective_rate = mutation_rate * (1 - fitness * 0.3)
        
        nodes = self.root.get_nodes()
        
        for node in nodes:
            if random.random() < effective_rate:
                # Protected Gene检查
                if node.protected:
                    if random.random() > 0.2:  # 80%概率跳过保护节点
                        continue
                
                # 选择变异类型（高fitness时偏向常数优化和微调）
                mutation_type = random.random()
                
                if fitness > 0.8:
                    # 高fitness：偏向常数优化和微调
                    if mutation_type < 0.4:
                        self._optimize_constant(node)
                    elif mutation_type < 0.7:
                        self._replace_node(node)
                    else:
                        self._add_branch(node) if random.random() < 0.5 else self._remove_branch(node)
                else:
                    # 低fitness：探索性变异
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
            # 替换操作（包含一元操作算子）
            all_ops = ['ADD', 'SUB', 'MUL', 'DIV', 'SIN', 'COS', 'EXP', 'LOG', 'NEG']
            old_op = node.op
            # 确保不会替换成同一个操作
            candidates = [o for o in all_ops if o != old_op]
            new_op = random.choice(candidates)
            node.op = new_op
            # 如果从一元变为二元，需要补充右子节点
            if old_op in OperationNode.UNARY_OPS and new_op not in OperationNode.UNARY_OPS:
                if node.right is None:
                    node.right = TreeGenome._generate_random_tree(1, is_left=False)
                    node.right.parent = node
            # 如果从二元变为一元，需要清空右子节点
            if old_op not in OperationNode.UNARY_OPS and new_op in OperationNode.UNARY_OPS:
                node.right = None
    
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
    
    @staticmethod
    def crossover(parent1, parent2):
        """TreeGenome 交叉：交换随机子树
        
        策略：
        - 从每个父本中随机选择一个操作节点
        - 交换这两个子树
        - 如果一方没有操作节点，则直接复制另一方
        """
        # 确保两个父本都有树结构
        if parent1.root is None and parent2.root is None:
            return TreeGenome(None)
        if parent1.root is None:
            return parent2.copy()
        if parent2.root is None:
            return parent1.copy()
        
        # 获取所有操作节点（排除叶节点）
        nodes1 = [n for n in parent1.root.get_nodes() if n.node_type == 'operation']
        nodes2 = [n for n in parent2.root.get_nodes() if n.node_type == 'operation']
        
        # 如果任一方没有操作节点，直接复制
        if not nodes1 or not nodes2:
            child = parent1.copy() if random.random() < 0.5 else parent2.copy()
            return child
        
        # 从每个父本随机选一个操作节点
        point1 = random.choice(nodes1)
        point2 = random.choice(nodes2)
        
        # 构建子代：复制 parent1，用 parent2 的子树替换对应节点
        child = parent1.copy()
        
        # 找到 child 中对应 point1 位置的节点
        child_nodes = child.root.get_nodes()
        child_op_nodes = [n for n in child_nodes if n.node_type == 'operation']
        
        if child_op_nodes:
            # 用 parent2 的子树替换第一个操作节点
            target = child_op_nodes[0]
            replacement = point2.copy()
            replacement.parent = target.parent
            
            if target.parent:
                if target.parent.left == target:
                    target.parent.left = replacement
                else:
                    target.parent.right = replacement
            else:
                child.root = replacement
        
        child.fitness = 0.0
        child.mdl_score = 0.0
        return child
    
    def __repr__(self):
        return f"TreeGenome(expr={self.to_expression()}, fitness={self.fitness:.4f}, mdl={self.mdl_score:.2f})"
    
    def __len__(self):
        return self.get_complexity()


# ============================================================================
# 向后兼容：线性 Genome 已移至 genome_linear.py
# ============================================================================
from .genome_linear import ArchiveMemory, Genome, GeneticOptimizer
