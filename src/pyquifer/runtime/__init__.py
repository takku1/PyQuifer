"""Runtime subpackage — tick loop, config, and diagnostics."""
from pyquifer.runtime.config import CycleConfig
from pyquifer.runtime.cycle import CognitiveCycle
from pyquifer.runtime.tick_result import (
    PROCESSING_MODE_BALANCED,
    PROCESSING_MODE_IMAGINATION,
    PROCESSING_MODE_NAMES,
    PROCESSING_MODE_PERCEPTION,
    TickResult,
)
