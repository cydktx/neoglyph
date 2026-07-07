import time
import sys
from collections import defaultdict


class Profiler:
    def __init__(self):
        self.instruction_counts = defaultdict(int)
        self.instruction_times = defaultdict(float)
        self.instruction_starts = {}
        self.memory_usage = 0
        self.tensor_stats = {
            'total_created': 0,
            'total_ops': 0,
            'avg_size': 0,
            'total_elements': 0
        }
        self.tape_size = 0
        self.error_count = 0
        self.op_sequence = []
        self.start_time = None
        self.total_time = 0
        self.enabled = True

    def start(self):
        if not self.enabled:
            return
        self.start_time = time.perf_counter()

    def stop(self, obj_map=None):
        if not self.enabled or self.start_time is None:
            return
        self.total_time = time.perf_counter() - self.start_time
        self._update_memory(obj_map)

    def _update_memory(self, obj_map=None):
        if hasattr(sys, 'getsizeof'):
            total = 0
            if obj_map is not None:
                for obj_id, tensor in obj_map.items():
                    total += sys.getsizeof(tensor.data)
            self.memory_usage = total

    def on_instruction_start(self, op_name):
        if not self.enabled:
            return
        self.instruction_starts[op_name] = time.perf_counter()
        self.op_sequence.append(op_name)

    def on_instruction_end(self, op_name):
        if not self.enabled:
            return
        if op_name in self.instruction_starts:
            elapsed = time.perf_counter() - self.instruction_starts[op_name]
            self.instruction_counts[op_name] += 1
            self.instruction_times[op_name] += elapsed
            del self.instruction_starts[op_name]

    def on_tensor_created(self, tensor):
        if not self.enabled:
            return
        self.tensor_stats['total_created'] += 1
        elements = tensor.data.size
        self.tensor_stats['total_elements'] += elements
        if self.tensor_stats['total_created'] > 0:
            self.tensor_stats['avg_size'] = (
                self.tensor_stats['total_elements'] / 
                self.tensor_stats['total_created']
            )

    def on_tensor_op(self):
        if not self.enabled:
            return
        self.tensor_stats['total_ops'] += 1

    def on_tape_record(self, tape_ops):
        if not self.enabled:
            return
        self.tape_size = len(tape_ops)

    def on_error(self):
        if not self.enabled:
            return
        self.error_count += 1

    def get_report(self):
        return {
            'instructions': dict(self.instruction_counts),
            'time': dict(self.instruction_times),
            'memory': self.memory_usage,
            'tensor_stats': self.tensor_stats,
            'tape_size': self.tape_size,
            'error_count': self.error_count,
            'total_time': self.total_time,
            'hot_instructions': self._find_hot_instructions(),
            'high_freq_combinations': self._find_high_freq_combinations(),
            'performance_bottlenecks': self._find_bottlenecks(),
            'instruction_efficiency': self._calculate_efficiency()
        }

    def _find_hot_instructions(self, top_n=5):
        sorted_ops = sorted(
            self.instruction_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [op for op, count in sorted_ops[:top_n]]

    def _find_high_freq_combinations(self, top_n=5):
        pairs = defaultdict(int)
        for i in range(len(self.op_sequence) - 1):
            pair = (self.op_sequence[i], self.op_sequence[i+1])
            pairs[pair] += 1
        sorted_pairs = sorted(
            pairs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [{'ops': pair, 'count': count} 
                for pair, count in sorted_pairs[:top_n]]

    def _find_bottlenecks(self, top_n=3):
        avg_times = {}
        for op, count in self.instruction_counts.items():
            if count > 0:
                avg_times[op] = self.instruction_times[op] / count
        sorted_ops = sorted(
            avg_times.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [{'op': op, 'avg_time_ms': time_ms * 1000} 
                for op, time_ms in sorted_ops[:top_n]]

    def _calculate_efficiency(self):
        total_ops = sum(self.instruction_counts.values())
        if total_ops == 0 or self.total_time == 0:
            return 0.0
        return total_ops / self.total_time

    def get_fitness_metrics(self, accuracy_weight=0.5, speed_weight=0.2, 
                           memory_weight=0.15, instruction_weight=0.15):
        report = self.get_report()
        
        speed_score = min(report['instruction_efficiency'] / 1000, 1.0) if report['instruction_efficiency'] > 0 else 0.0
        
        memory_score = 1.0 - min(report['memory'] / (1024 * 1024), 1.0)
        
        instruction_score = 0.0
        if report['instructions']:
            total = sum(report['instructions'].values())
            instruction_score = 1.0 - min(total / 10000, 1.0)
        
        error_score = 1.0 - min(report['error_count'] / 10, 1.0)
        
        return {
            'speed': speed_score,
            'memory': memory_score,
            'instruction': instruction_score,
            'error': error_score,
            'weights': {
                'accuracy': accuracy_weight,
                'speed': speed_weight,
                'memory': memory_weight,
                'instruction': instruction_weight
            }
        }

    def reset(self):
        self.instruction_counts = defaultdict(int)
        self.instruction_times = defaultdict(float)
        self.instruction_starts = {}
        self.memory_usage = 0
        self.tensor_stats = {
            'total_created': 0,
            'total_ops': 0,
            'avg_size': 0,
            'total_elements': 0
        }
        self.tape_size = 0
        self.error_count = 0
        self.op_sequence = []
        self.start_time = None
        self.total_time = 0

    def __repr__(self):
        return f"Profiler(ops={sum(self.instruction_counts.values())}, " \
               f"time={self.total_time:.4f}s, memory={self.memory_usage}B)"


class ProfilerMixin:
    def __init__(self):
        self.profiler = Profiler()

    def set_profiler(self, profiler):
        self.profiler = profiler

    def get_profiler(self):
        return self.profiler
