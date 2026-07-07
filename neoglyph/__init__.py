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
from .evolution import (
    EvolutionEngine,
    TreeEvolutionEngine,
    SequentialEvaluator,
    ParallelEvaluator,
    TournamentSelector,
    FitnessScorer,
    DiscoveryScorer,
    EarlyStopper,
    FitnessSharing,
    IslandModel,
)
from .evolution_advanced import (
    DiscoveryScore,
    InvalidProgramFilter,
    CurriculumEvolution,
    AdvancedEvolutionEngine
)
from .applications import (
    BaseApplication,
    SymbolicRegressor,
    PhysicsDiscoverer,
    ParallelSymbolicRegressor,
)

__version__ = "4.0"
