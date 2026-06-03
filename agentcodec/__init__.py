"""
AgentCodec — communication-theoretic reliability for LLM agents.

Source-available under the PolyForm Noncommercial License 1.0.0. See
LICENSE and COMMERCIAL.md for terms.

Two surfaces are exposed:

1. **Library API** — for embedding reliability into a production app:

       from agentcodec import ReliabilityModule
       mod = ReliabilityModule.from_yaml("reliability.yaml")
       result = mod.run("What's the capital of France?", category="qa")

2. **Benchmark API** — preserved for paper reproduction:

       from agentcodec.runner import BenchmarkRunner, ExperimentConfig

SemKNN routing is a remote service in this release. The trained artifacts
are not redistributed; the client sends only a unit-norm embedding (no
prompt) to the SemKNN backend. See the README §"Privacy & data flow".
"""

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("agentcodec")
except Exception:
    __version__ = "0.3.0"

# --- `.env` support (NOT auto-loaded at library level) -----------------
# The library follows the OpenAI / Anthropic SDK convention: read keys
# and process-wide knobs from `os.environ` only. Loading `.env` is the
# caller's responsibility (it would be surprising for an embedded
# library to mutate the host process's environment on import).
#
# `load_dotenv` is exported so users / scripts that DO want `.env`
# autoloading can opt in with one line. The `examples/` scripts call
# this automatically via `examples/_common.py`.
from ._dotenv import load_dotenv

# --- Library API ---
from .api import ReliabilityModule

# --- Underlying primitives, useful for advanced integrators ---
from .channel import MODEL_COSTS, AgentChannel, QualityScorer
from .config import (
    CISCConfig,
    CostPer1M,
    CriticConfig,
    Defaults,
    FixedStrategy,
    JudgeConfig,
    LibraryConfig,
    ModelConfig,
    RoutedStrategy,
    RouterConfig,
    SoftNormalization,
    StreamingDefaults,
    TelemetryYAMLConfig,
    ThinkingConfig,
)
from .cost import CostBreakdown, CostSource
from .dispatch import KNOWN_TECHNIQUES, DispatchContext, dispatch
from .evaluation import (
    ConfigStats,
    EvalReport,
    Evaluator,
    PairwiseComparison,
)
from .messages import (
    ChatRequest,
    ChatResponse,
    ContentBlock,
    ImageBlock,
    Message,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    ToolUseBlock,
)
from .models import (
    AgentOutput,
    CombiningStrategy,
    HARQMode,
    ReliabilityRun,
    TaskCategory,
    TaskItem,
)
from .presets import KNOWN_PRESETS, build_preset_config
from .results import (
    Event,
    FinalEvent,
    ProgressEvent,
    ReliabilityResult,
    TokenEvent,
    WarningEvent,
)
from .routing import (
    ACMTableRouter,
    AutoCategoryClassifier,
    FixedRouter,
    LinearRouter,
    RemoteSemKNNRouter,
    Router,
    RouterDecision,
    canonical_family,
)
from .settings import Settings
from .telemetry import Telemetry, TelemetryConfig

__all__ = [
    # Presets
    "KNOWN_PRESETS",
    "KNOWN_TECHNIQUES",
    "MODEL_COSTS",
    "ACMTableRouter",
    # Primitives
    "AgentChannel",
    "AgentOutput",
    "AutoCategoryClassifier",
    "CISCConfig",
    # Provider-neutral chat types (used by the compat shims)
    "ChatRequest",
    "ChatResponse",
    "CombiningStrategy",
    "ConfigStats",
    "ContentBlock",
    # Cost transparency
    "CostBreakdown",
    "CostPer1M",
    "CostSource",
    "CriticConfig",
    "Defaults",
    "DispatchContext",
    "EvalReport",
    # Evaluation
    "Evaluator",
    "Event",
    "FinalEvent",
    "FixedRouter",
    "FixedStrategy",
    "HARQMode",
    "ImageBlock",
    "JudgeConfig",
    # Config
    "LibraryConfig",
    "LinearRouter",
    "Message",
    "ModelConfig",
    "PairwiseComparison",
    "ProgressEvent",
    "QualityScorer",
    # API
    "ReliabilityModule",
    # Results
    "ReliabilityResult",
    "ReliabilityRun",
    "RemoteSemKNNRouter",
    "RoutedStrategy",
    # Routing
    "Router",
    "RouterConfig",
    "RouterDecision",
    "Settings",
    "SoftNormalization",
    "StreamingDefaults",
    "TaskCategory",
    "TaskItem",
    # Telemetry
    "Telemetry",
    "TelemetryConfig",
    "TelemetryYAMLConfig",
    "TextBlock",
    "ThinkingConfig",
    "TokenEvent",
    "ToolCall",
    "ToolResultBlock",
    "ToolUseBlock",
    "WarningEvent",
    "__version__",
    "build_preset_config",
    "canonical_family",
    "dispatch",
    "load_dotenv",
]
