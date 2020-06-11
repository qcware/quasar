"""
Build and execute quantum circuits on various backends.

Quasar is a package to simplify the building and execution of quantum computing
circuits for quantum computing in a reasonably vendor-agnostic way.
It provides backends for multiple providers, but also has a credible
classical simulator.  It is also the basis for circuit-model functionality
in QCWare's Forge platform (http://www.qcware.com)
"""
__version__ = '1.0.0'
from .circuit import Matrix, Gate, Circuit
from .circuit import ControlledGate, CompositeGate
from .pauli import PauliStarter, PauliOperator, PauliString, Pauli, PauliExpectation
from .pauli import PauliOperator
from .index_allocator import IndexAllocator, NegativeIndexAllocator
from .transpiler import Transpiler
from .backend import Backend
from .quasar_backend import QuasarSimulatorBackend
from .measurement import Histogram, ProbabilityHistogram, CountHistogram
from .format import format_statevector
from .davidson import Davidson, run_davidson
from .parameters import ParameterGroup, FixedParameterGroup, IdentityParameterGroup, LinearParameterGroup, CompositeParameterGroup
from .observable import VariationalObservable
from .optimizer import Optimizer, BFGSOptimizer
from .options import Options
