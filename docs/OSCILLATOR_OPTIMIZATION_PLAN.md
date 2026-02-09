# Oscillator Optimization Plan — Tasks #42 & #43

**Status:** Planned (not yet implemented)
**Priority:** Low — oscillators are <0.2ms of tick() time at N=32-64
**When it matters:** N>256 oscillators, batched multi-workspace, GPU-resident pipelines

---

## Task #42: Sparse Topology Fast Path for Kuramoto

### Problem

Non-global topologies (small_world, scale_free, ring, learnable) use dense O(N²) computation:

```python
# oscillators.py:308 — current implementation
phase_diffs = phases.unsqueeze(1) - phases.unsqueeze(0)  # (n, n) dense matrix
weighted_interaction = effective_adj * torch.sin(phase_diffs)
interaction_sum = torch.sum(weighted_interaction, dim=1)
```

For a ring topology with k=4 neighbors, this computes N² phase differences when only 2k*N are needed.

### Solution: CSR Neighbor Iteration

Based on the multilayer-cudamoto CSR pattern (researched in earlier session).

**Storage change:**
```python
# In _init_adjacency(), convert dense adj to CSR:
self.register_buffer('_csr_row_ptr', row_ptr)   # (N+1,) int
self.register_buffer('_csr_col_idx', col_idx)    # (nnz,) int
self.register_buffer('_csr_values', values)       # (nnz,) float
```

**Forward pass change:**
```python
# Replace dense path with sparse gather:
neighbor_phases = phases[self._csr_col_idx]              # (nnz,)
self_phases = phases.repeat_interleave(
    self._csr_row_ptr[1:] - self._csr_row_ptr[:-1]
)                                                         # (nnz,)
phase_diffs = neighbor_phases - self_phases               # (nnz,)
weighted = self._csr_values * torch.sin(phase_diffs)     # (nnz,)
interaction_sum = torch.zeros(N, device=phases.device)
interaction_sum.scatter_add_(0,
    torch.arange(N, device=phases.device).repeat_interleave(
        self._csr_row_ptr[1:] - self._csr_row_ptr[:-1]
    ),
    weighted
)
```

**Precision-weighted variant:**
```python
# CSR values become adj_weight * source_precision
effective_values = self._csr_values * self.precision[self._csr_col_idx].detach()
```

### Files to Modify
- `oscillators.py`: `_init_adjacency()` (add CSR buffers), `forward()` (sparse path), `get_adjacency()` (reconstruct dense for compatibility), RK4 closure
- `tests/test_oscillators.py`: Add CSR correctness tests for each topology

### Complexity
- ~100 lines of new code
- Must handle learnable topology (CSR rebuild on adjacency_logits change)
- RK4 closure needs the same CSR path (currently recomputes coupling)
- `compute_attractor_stability()` and `compute_local_stats()` also use adjacency

### Expected Speedup
- Ring k=4, N=256: ~32x fewer multiply-adds (2kN vs N²)
- Ring k=4, N=64: ~8x fewer, but absolute time is already <0.05ms
- Global topology: NO CHANGE (already O(N), no adjacency matrix)
- Learnable topology: Minimal benefit (still effectively dense)

---

## Task #43: Fused Oscillator Fast Path

### Problem

The current forward pass creates multiple intermediate tensors:

```python
sin_phases = torch.sin(phases)       # alloc 1
cos_phases = torch.cos(phases)       # alloc 2
interaction_sum = ...                 # alloc 3
normalized_interaction = ...          # alloc 4
dphi = omega + K * normalized + ext   # alloc 5
phases = phases + dt * dphi           # alloc 6
```

Then `compute_order_parameter()` recomputes sin/cos:
```python
sin_sum = torch.sin(phases).sum()    # redundant sin
cos_sum = torch.cos(phases).sum()    # redundant cos
```

### Solution: Single-Pass Fused Function

```python
def _fused_kuramoto_step(
    phases: Tensor, omega: Tensor, K: Tensor, dt: float,
    external: Optional[Tensor] = None,
) -> Tuple[Tensor, float, float]:
    """Fused phase update + order parameter in one pass.
    Returns (new_phases, R, Psi)."""
    sin_p = torch.sin(phases)
    cos_p = torch.cos(phases)

    # Order parameter (reuse sin/cos)
    mean_sin = sin_p.mean()
    mean_cos = cos_p.mean()
    R = torch.sqrt(mean_sin**2 + mean_cos**2)
    Psi = torch.atan2(mean_sin, mean_cos)

    # Kuramoto coupling (global topology)
    sum_sin = sin_p.sum()
    sum_cos = cos_p.sum()
    interaction = (cos_p * (sum_sin - sin_p) - sin_p * (sum_cos - cos_p)) / max(N-1, 1)

    # Phase update
    dphi = omega + K * interaction
    if external is not None:
        dphi = dphi + external
    new_phases = (phases + dt * dphi).remainder_(2 * math.pi)

    return new_phases, R, Psi
```

This eliminates:
- Redundant sin/cos in `compute_order_parameter()`
- 2-3 intermediate tensor allocations
- Second pass over phases array

### torch.compile Synergy

This fused function is an ideal `torch.compile` target — pure tensor ops, no branching, no `.item()`. With inductor, the 6 elementwise ops fuse into 1-2 kernels.

### Files to Modify
- `oscillators.py`: Add `_fused_kuramoto_step()`, use in `forward()` for global topology + euler integration
- RK4 path stays separate (requires 4 evaluations, can't easily fuse)
- `integration.py`: Update tick() to read R/Psi from forward return instead of separate call

### Complexity
- ~60 lines of new code
- Must preserve backward compatibility (existing code reads `compute_order_parameter()` separately)
- RK4 integration method can't use fused path (needs intermediate evaluations)
- Precision-weighted variant needs its own fused version

### Expected Speedup
- Eliminates ~2 redundant sin/cos calls per tick (small absolute savings at N=64)
- Reduces intermediate tensor count from 6 to 3
- Main benefit is torch.compile fusion potential, not raw Python speedup

---

## Implementation Order

1. **#43 first** — simpler, self-contained, enables torch.compile fusion
2. **#42 second** — more invasive, requires CSR storage migration

## Decision: When to Implement

These optimizations become worthwhile when:
- [ ] N > 256 oscillators (multi-workspace cross-bleed scenarios)
- [ ] GPU-resident pipeline where kernel launch overhead matters
- [ ] Batched inference with multiple CognitiveCycles

At current N=32-64 on CPU, the oscillator forward pass is already <0.2ms and is NOT the bottleneck. The HPC (hierarchical predictive coding) and diagnostics dict construction dominate tick() time.
