"""
PyQuifer test configuration.

Patches thinc's global numpy-seed hook so that pytest-randomly's CRC32-derived
seeds (which can exceed numpy's u32 range) don't cause ValueError at every
test setup/teardown.

Root cause: pytest-randomly v4+ computes `(_crc32(nodeid) ± 1) % 2**32`, which
can produce 2**32 exactly; thinc.util.fix_random_seed then calls
numpy.random.seed(2**32) → ValueError: Seed must be between 0 and 2**32 - 1.
"""
try:
    import thinc.util as _thinc_util
    _thinc_util.fix_random_seed = lambda seed: None  # type: ignore[method-assign]
except ImportError:
    pass
