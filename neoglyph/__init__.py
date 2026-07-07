from .tensor import Tensor
from .vm import NeoGlyphVM
from .genome import Genome, GeneticOptimizer
from .genome import (
    ProgramNode,
    ConstantNode,
    VariableNode,
    OperationNode,
    TreeGenome,
    ArchiveMemory
)
from .profiler import Profiler
from .evolution import EvolutionEngine
from .evolution_advanced import (
    DiscoveryScore,
    InvalidProgramFilter,
    ParallelEvaluator,
    CurriculumEvolution,
    AdvancedEvolutionEngine
)

__version__ = "3.2"
