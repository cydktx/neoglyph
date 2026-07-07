"""线性 Genome 模块 - 已移至独立文件，保持向后兼容"""
import numpy as np
import random
from .vm import NeoGlyphVM

_OP_CODES = {
    'PUSH': 1, 'ADD': 2, 'SUB': 3, 'MUL': 4, 'DIV': 5,
    'SHAPE': 6, 'POP': 7, 'STORE': 8, 'LOAD': 9, 'PRINT': 10,
    'TAPE': 11, 'UNTAPE': 12, 'GRAD': 13, 'HALT': 14, 'JMP': 15,
    'JMP_IF': 16, 'RELU': 17, 'NEG': 18, 'POW': 19, 'MATMUL': 20,
    'SIN': 21, 'COS': 22, 'EXP': 23, 'LOG': 24,
}

_CODE_OPS = {v: k for k, v in _OP_CODES.items()}

_VAR_PREFIX = 100


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