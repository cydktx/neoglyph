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
    ParetoFront,
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
from .visualization import (
    plot_fit_curve,
    plot_evolution_history,
    plot_expression_tree,
    plot_pareto_front,
    print_tree,
)

__version__ = "4.1"
