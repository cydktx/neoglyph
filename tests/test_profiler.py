import unittest
from neoglyph.vm import NeoGlyphVM
from neoglyph.profiler import Profiler


class TestProfiler(unittest.TestCase):
    def test_instruction_counts(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 2
        STORE a
        PUSH 3
        STORE b
        LOAD a
        LOAD b
        ADD
        PRINT
        HALT
        """
        vm.run(script)
        report = vm.get_profile_report()
        self.assertIn('PUSH', report['instructions'])
        self.assertIn('ADD', report['instructions'])
        self.assertEqual(report['instructions']['PUSH'], 2)
        self.assertEqual(report['instructions']['ADD'], 1)

    def test_time_stats(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 1
        PUSH 2
        ADD
        HALT
        """
        vm.run(script)
        report = vm.get_profile_report()
        self.assertGreater(report['total_time'], 0)
        self.assertIn('ADD', report['time'])

    def test_vm_not_broken(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 10
        STORE a
        PUSH 5
        STORE b
        LOAD a
        LOAD b
        ADD
        PRINT
        HALT
        """
        vm.run(script)
        self.assertEqual(len(vm.stack), 0)
        self.assertIn('a', vm.vars)

    def test_tensor_stats(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 1 2 3
        STORE x
        PUSH 4 5 6
        STORE y
        LOAD x
        LOAD y
        ADD
        STORE z
        HALT
        """
        vm.run(script)
        report = vm.get_profile_report()
        self.assertGreater(report['tensor_stats']['total_created'], 0)
        self.assertGreater(report['tensor_stats']['total_ops'], 0)

    def test_memory_usage(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 1 2 3 4
        STORE a
        HALT
        """
        vm.run(script)
        report = vm.get_profile_report()
        self.assertGreaterEqual(report['memory'], 0)

    def test_hot_instructions(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 1
        PUSH 2
        ADD
        PUSH 3
        ADD
        PUSH 4
        ADD
        HALT
        """
        vm.run(script)
        report = vm.get_profile_report()
        hot = report['hot_instructions']
        self.assertIn('PUSH', hot)
        self.assertIn('ADD', hot)

    def test_high_freq_combinations(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 1
        STORE a
        PUSH 2
        STORE b
        PUSH 3
        STORE c
        HALT
        """
        vm.run(script)
        report = vm.get_profile_report()
        combos = report['high_freq_combinations']
        self.assertGreater(len(combos), 0)

    def test_bottlenecks(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 1
        PUSH 2
        ADD
        HALT
        """
        vm.run(script)
        report = vm.get_profile_report()
        bottlenecks = report['performance_bottlenecks']
        self.assertGreaterEqual(len(bottlenecks), 0)

    def test_error_count(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 1
        INVALID_OP
        HALT
        """
        try:
            vm.run(script)
        except ValueError:
            pass
        report = vm.get_profile_report()
        self.assertEqual(report['error_count'], 1)

    def test_fitness_metrics(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 2
        PUSH 3
        ADD
        HALT
        """
        vm.run(script)
        metrics = vm.get_fitness_metrics()
        self.assertIn('speed', metrics)
        self.assertIn('memory', metrics)
        self.assertIn('instruction', metrics)
        self.assertIn('error', metrics)
        self.assertIn('weights', metrics)

    def test_profiler_enabled_disabled(self):
        profiler = Profiler()
        profiler.enabled = False
        profiler.on_instruction_start('ADD')
        profiler.on_instruction_end('ADD')
        self.assertEqual(profiler.instruction_counts['ADD'], 0)

    def test_profiler_reset(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 1
        PUSH 2
        ADD
        HALT
        """
        vm.run(script)
        vm.profiler.reset()
        report = vm.get_profile_report()
        self.assertEqual(sum(report['instructions'].values()), 0)
        self.assertEqual(report['total_time'], 0)


class TestProfilerIntegration(unittest.TestCase):
    def test_gradients_with_profiler(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 2
        STORE a
        PUSH 3
        STORE b

        TAPE
        LOAD a
        LOAD b
        ADD
        STORE c
        UNTAPE

        GRAD

        LOAD a
        PRINT
        LOAD b
        PRINT
        HALT
        """
        vm.run(script)
        report = vm.get_profile_report()
        self.assertIn('GRAD', report['instructions'])
        self.assertGreater(report['tape_size'], 0)

    def test_complex_script(self):
        vm = NeoGlyphVM()
        script = """
        PUSH 1
        STORE x
        PUSH 2
        STORE y
        PUSH 3
        STORE z

        TAPE
        LOAD x
        LOAD y
        ADD
        LOAD z
        MUL
        STORE result
        UNTAPE

        GRAD

        LOAD result
        PRINT
        HALT
        """
        vm.run(script)
        report = vm.get_profile_report()
        self.assertIn('ADD', report['instructions'])
        self.assertIn('MUL', report['instructions'])
        self.assertIn('GRAD', report['instructions'])


if __name__ == '__main__':
    unittest.main()
