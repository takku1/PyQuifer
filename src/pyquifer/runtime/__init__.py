"""Runtime subpackage — tick loop, config, and diagnostics."""
from pyquifer.runtime.tick_result import (
    TickResult, PROCESSING_MODE_PERCEPTION, PROCESSING_MODE_IMAGINATION,
    PROCESSING_MODE_BALANCED, PROCESSING_MODE_NAMES,
)
from pyquifer.runtime.config import CycleConfig
from pyquifer.runtime.cycle import CognitiveCycle
