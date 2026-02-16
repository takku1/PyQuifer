# Enhancement A: MCP-as-Organ Protocol — Detailed Implementation Plan

**Status:** Planning
**Priority:** Highest (recommended first by neuroscience blueprint)
**Estimated scope:** 1 new PyQuifer file, 1 new Mizuki runtime loop, 4 file modifications, ~350 lines new code

---

## Goal

Make each MCP server a first-class workspace competitor in PyQuifer's global workspace. When Mizuki connects to GitHub, filesystem, Slack, etc. via MCP, those processes should compete for attention alongside internal organs (HPC, Motivation, Selection) through the same salience → ignition → broadcast pipeline.

**End state:** An MCP resource that detects something urgent (e.g., Slack mention, file change, GitHub PR review request) can win workspace competition and redirect Mizuki's attention — just like a biological sensory signal can interrupt ongoing thought.

---

## Current Architecture (What Exists)

### PyQuifer Side

**`organ.py`** — Complete organ framework:
- `Organ` ABC with `observe()`, `propose()`, `accept()` contract
- `Proposal(content: Tensor, salience: float, tags: Set[str], cost: float, organ_id: str)`
- `OscillatoryWriteGate` — CTC (Communication Through Coherence) phase gating
- `PreGWAdapter` — Projects organ latents ↔ workspace dimension
- Each organ has `phase`, `frequency`, `standing_latent` buffers

**`global_workspace.py`** — Complete competition:
- `SalienceComputer` → `IgnitionDynamics` → `CompetitionDynamics` → `GlobalBroadcast`
- `WorkspaceEnsemble` with `StandingBroadcast` + `CrossBleedGate`
- `DiversityTracker` (anti-collapse, boosts underrepresented organs)

**`integration.py`** — Wiring:
- `CognitiveCycle.register_organ(organ, adapter=None)` appends to `_organs`
- `_run_workspace_competition()` runs full observe → propose → gate → compete → broadcast loop
- Step 8b of `tick()`

**`bridge.py`** — External API:
- `ModulationState` returned by `step()` — does NOT currently expose organ/workspace state

### Mizuki Side

**`mcp_adapter.py`** — MCP client:
- `MCPClient` manages `ClientSessionGroup` (multiple MCP servers)
- `ClientSession` wraps one transport (stdio or in-memory)
- Sessions have `_resources` (cached) and `_tools` (cached)
- `call_tool(name, args, timeout=60)` — synchronous JSON-RPC
- `read_resource(uri)` — fetch resource state

**`mcp_server.py`** — Mizuki's own MCP server:
- 8 resources: oscillators/state, consciousness/metrics, emotional/state, homeostasis/drives, memory/status, hardware/health, somatic/feelings, attention/focus
- 6 tools: adjust_temperature, trigger_consolidation, adjust_coherence_target, reset_homeostasis, focus_attention, inject_somatic_signal

**Current MCP flow:** MCP tools are called **synchronously on-demand** when the LLM or actor requests a tool. No proactive polling. No organ wrapper. No workspace competition.

---

## Implementation Plan

### File 1: NEW `PyQuifer/src/pyquifer/mcp_organ.py` (~120 lines)

```python
"""MCP-as-Organ: wraps an external MCP resource as a workspace-competing organ."""

class MCPOrganConfig:
    """Configuration for one MCP organ instance."""
    organ_id: str           # e.g., "mcp:github", "mcp:filesystem"
    resource_uri: str       # e.g., "github://notifications", "fs://watched"
    workspace_dim: int      # Must match CognitiveCycle workspace_dim
    latent_dim: int         # Native dimension of this organ's state
    base_salience: float    # Resting salience (0.0-1.0)
    salience_decay: float   # How fast salience decays when nothing changes (default 0.95)
    cost: float             # Computational cost estimate for this organ (default 0.1)
    tags: Set[str]          # Domain tags, e.g., {"external", "github", "code"}
    poll_stale_after: float # Seconds before resource state considered stale (default 5.0)

class MCPOrgan(Organ):
    """Wraps an MCP resource endpoint as a PyQuifer organ.

    Lifecycle per tick:
    1. observe() — reads cached resource state, encodes to latent
    2. propose() — computes salience from state change, returns Proposal
    3. accept() — updates standing latent from broadcast winner

    Resource polling happens OUTSIDE the tick loop (async, at ~1 Hz).
    The organ reads from a pre-fetched cache, never blocks the tick.
    """

    def __init__(self, config: MCPOrganConfig):
        super().__init__(
            organ_id=config.organ_id,
            latent_dim=config.latent_dim,
            workspace_dim=config.workspace_dim,
        )
        self.config = config

        # State cache (written by async poller, read by observe)
        self._cached_state: Optional[torch.Tensor] = None
        self._prev_state: Optional[torch.Tensor] = None
        self._state_timestamp: float = 0.0
        self._change_magnitude: float = 0.0  # How much state changed

        # Encoder: raw MCP state → latent_dim
        # Simple linear + tanh, trainable via cycle consistency
        self._encoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.Tanh(),
        )

    def update_cache(self, state_tensor: torch.Tensor, timestamp: float):
        """Called by async poller (NOT during tick). Thread-safe via tensor copy."""
        self._prev_state = self._cached_state
        self._cached_state = state_tensor.clone().detach()
        self._state_timestamp = timestamp

        # Compute change magnitude for salience
        if self._prev_state is not None:
            self._change_magnitude = (state_tensor - self._prev_state).norm().item()
        else:
            self._change_magnitude = self.config.base_salience

    def observe(self, sensory_input: torch.Tensor, global_broadcast: Optional[torch.Tensor] = None):
        """Read cached state into local latent. Never blocks."""
        if self._cached_state is not None:
            self._latent = self._encoder(self._cached_state)
        # Also incorporate broadcast if available
        if global_broadcast is not None:
            self.update_standing(global_broadcast)

    def propose(self, gw_state: Optional[torch.Tensor] = None) -> Proposal:
        """Compute salience from state change magnitude + staleness."""
        if self._latent is None:
            # No state yet — minimal proposal
            return Proposal(
                content=self.standing_latent,
                salience=0.0,
                tags=self.config.tags,
                cost=self.config.cost,
                organ_id=self.config.organ_id,
            )

        # Salience = base + change_magnitude, decayed by staleness
        import time
        staleness = time.time() - self._state_timestamp
        freshness = max(0.0, 1.0 - staleness / self.config.poll_stale_after)

        salience = (
            self.config.base_salience * 0.3
            + self._change_magnitude * 0.7
        ) * freshness

        return Proposal(
            content=self._latent,
            salience=float(salience),
            tags=self.config.tags,
            cost=self.config.cost,
            organ_id=self.config.organ_id,
        )

    def accept(self, global_broadcast: torch.Tensor):
        """Update standing latent from broadcast."""
        self.update_standing(global_broadcast)
```

**Key design decisions:**
- **Non-blocking observe():** Reads from pre-fetched cache. The async poller writes to `_cached_state`; the tick loop reads it. No I/O during tick.
- **Salience from change detection:** Big state change = high salience (something happened). Stale state = decaying salience (nothing new).
- **Simple encoder:** Linear + tanh. Trainable via existing `PreGWAdapter` cycle consistency loss.
- **Thread safety:** `update_cache()` clones tensors. Worst case: tick reads slightly stale state (one poll behind). Acceptable.

---

### File 2: NEW `src/mizuki/runtime/loops/mcp_organ_loop.py` (~100 lines)

```python
"""Async loop that polls MCP resources and feeds MCPOrgan caches."""

class MCPOrganLoop:
    """Polls MCP resources at ~1 Hz, converts to tensors, updates organ caches.

    Runs as an asyncio task alongside OscillatorLoop (5 Hz) and CognitiveLoop (3 Hz).
    Much slower than internal loops because MCP I/O is expensive.
    """

    def __init__(self, organs: List[MCPOrgan], mcp_client: MCPClient, hz: float = 1.0):
        self.organs = organs
        self.mcp_client = mcp_client
        self.interval = 1.0 / hz
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    async def _loop(self):
        while self._running:
            t0 = time.time()
            await self._poll_all()
            elapsed = time.time() - t0
            await asyncio.sleep(max(0, self.interval - elapsed))

    async def _poll_all(self):
        """Poll all MCP resources concurrently."""
        tasks = [self._poll_one(organ) for organ in self.organs]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _poll_one(self, organ: MCPOrgan):
        """Read one MCP resource, encode to tensor, update organ cache."""
        try:
            result = await organ.mcp_session.read_resource(organ.config.resource_uri)
            tensor = self._encode_resource(result, organ.config.latent_dim)
            organ.update_cache(tensor, time.time())
        except Exception as e:
            logger.debug(f"MCP poll failed for {organ.config.organ_id}: {e}")

    def _encode_resource(self, resource_data: Any, dim: int) -> torch.Tensor:
        """Convert MCP resource JSON → fixed-dim tensor.

        Strategy: hash-based feature encoding for arbitrary JSON.
        Numeric values → direct, strings → hash embedding, nested → flatten.
        """
        # Extract numeric features from resource data
        features = self._flatten_to_numbers(resource_data)

        # Pad or truncate to dim
        if len(features) >= dim:
            return torch.tensor(features[:dim], dtype=torch.float32)
        else:
            padded = features + [0.0] * (dim - len(features))
            return torch.tensor(padded, dtype=torch.float32)

    def _flatten_to_numbers(self, data, prefix="") -> List[float]:
        """Recursively flatten JSON to float list."""
        result = []
        if isinstance(data, (int, float)):
            result.append(float(data))
        elif isinstance(data, str):
            # Hash string to a float in [-1, 1]
            h = hash(data) % (2**31)
            result.append(h / (2**31) * 2 - 1)
        elif isinstance(data, dict):
            for k, v in sorted(data.items()):
                result.extend(self._flatten_to_numbers(v, f"{prefix}.{k}"))
        elif isinstance(data, (list, tuple)):
            for i, v in enumerate(data):
                result.extend(self._flatten_to_numbers(v, f"{prefix}[{i}]"))
        elif isinstance(data, bool):
            result.append(1.0 if data else 0.0)
        return result
```

**Key design decisions:**
- **1 Hz default:** MCP I/O is expensive (JSON-RPC over stdio). 1 Hz is plenty for external state changes.
- **Concurrent polling:** All organs polled simultaneously via `asyncio.gather`.
- **Hash-based encoding:** Converts arbitrary JSON resource state → fixed-dim tensor. Simple but sufficient — the organ's `_encoder` network learns the useful mapping.
- **Failure tolerance:** Individual poll failures are logged and skipped. The organ uses its last cached state (with decaying freshness).

---

### File 3: MODIFY `PyQuifer/src/pyquifer/__init__.py` (~3 lines)

Add exports:
```python
from .mcp_organ import MCPOrgan, MCPOrganConfig
```

---

### File 4: MODIFY `PyQuifer/src/pyquifer/integration.py` (~15 lines)

In `_run_workspace_competition()` return dict, add organ standings:

```python
# After competition resolves, collect organ standings
organ_standings = {}
for organ, _adapter in self._organs:
    organ_standings[organ.organ_id] = organ.standing_latent.clone()

result["organ_standings"] = organ_standings
result["workspace_winner_id"] = winner_organ_id  # Already available from competition
```

Add to `CycleConfig`:
```python
use_mcp_organs: bool = False  # Enable MCP organ integration
mcp_organ_latent_dim: int = 64  # Default latent dim for MCP organs
```

---

### File 5: MODIFY `PyQuifer/src/pyquifer/bridge.py` (~20 lines)

Extend `ModulationState` to expose organ/workspace info:

```python
@dataclass (or add fields to existing)
class ModulationState:
    # ... existing fields ...

    # NEW: Organ/workspace state (optional, only when use_mcp_organs=True)
    workspace_winner_id: Optional[str] = None      # Which organ won this tick
    workspace_broadcast: Optional[torch.Tensor] = None  # Winner's content
    organ_standings: Optional[Dict[str, torch.Tensor]] = None  # All standing latents
```

In `step()`, populate from tick diagnostics when workspace is active.

---

### File 6: MODIFY `src/mizuki/core/brain_unified.py` (~30 lines)

In `_init_mcp_discovery()` or a new `_init_mcp_organs()`:

```python
def _init_mcp_organs(self):
    """Create MCPOrgan wrappers for connected MCP servers."""
    if not self._mcp_client or not self._bridge:
        return

    organs = []
    for session_name, session in self._mcp_client.group.sessions.items():
        # Create one organ per MCP resource
        for resource_meta in session._resources.values():
            config = MCPOrganConfig(
                organ_id=f"mcp:{session_name}:{resource_meta.name}",
                resource_uri=resource_meta.uri,
                workspace_dim=self._bridge.cycle.workspace_dim,
                latent_dim=64,
                base_salience=0.2,
                tags={"external", "mcp", session_name},
            )
            organ = MCPOrgan(config)
            organ.mcp_session = session  # Store ref for poller
            self._bridge.cycle.register_organ(organ)
            organs.append(organ)

    self._mcp_organs = organs
    return organs
```

---

### File 7: MODIFY `src/mizuki/runtime/runtime.py` (~20 lines)

Add `MCPOrganLoop` to the runtime's concurrent loops:

```python
# In CognitiveRuntime.__init__ or start():
if brain._mcp_organs:
    from .loops.mcp_organ_loop import MCPOrganLoop
    self.mcp_organ_loop = MCPOrganLoop(
        organs=brain._mcp_organs,
        mcp_client=brain._mcp_client,
        hz=1.0,
    )
    # Start alongside other loops
    await self.mcp_organ_loop.start()
```

---

## Data Flow (End to End)

```
External MCP Servers (GitHub, Slack, filesystem, etc.)
    ↓ JSON-RPC (1 Hz poll)
MCPOrganLoop._poll_all()
    ↓ torch.Tensor (hash-encoded)
MCPOrgan.update_cache(tensor, timestamp)
    ↓ (cached, read by tick)
CognitiveCycle.tick() → Step 8b: _run_workspace_competition()
    ├─ MCPOrgan.observe()  → reads cache → encodes to latent
    ├─ MCPOrgan.propose()  → salience from change + freshness
    ├─ OscillatoryWriteGate → phase coherence gating
    ├─ DiversityTracker    → anti-collapse boost
    ├─ GlobalWorkspace     → competition with HPC, Motivation, Selection organs
    ├─ Winner selected     → broadcast to ALL organs
    └─ MCPOrgan.accept()   → updates standing latent
    ↓
ModulationState (includes workspace_winner_id, broadcast, standings)
    ↓
Mizuki brain/runtime consumes winner info
    ↓ (if winner is MCP organ)
Route attention to that MCP server's domain
```

---

## Test Plan

### Unit Tests (`PyQuifer/tests/test_mcp_organ.py`)

1. **`test_mcp_organ_propose_no_state`** — Organ with no cached state returns salience=0
2. **`test_mcp_organ_propose_with_change`** — Large state change → high salience
3. **`test_mcp_organ_staleness_decay`** — Old cached state → decaying salience
4. **`test_mcp_organ_observe_encode`** — observe() produces correct-dim latent
5. **`test_mcp_organ_accept_standing`** — accept() updates standing latent EMA
6. **`test_mcp_organ_phase_coupling`** — step_oscillator() couples to global phase
7. **`test_mcp_organ_registers_in_cycle`** — register_organ() accepts MCPOrgan
8. **`test_mcp_organ_competes_in_workspace`** — MCPOrgan participates in full competition round with HPCOrgan
9. **`test_mcp_organ_wins_high_salience`** — MCPOrgan with high salience beats low-salience internal organ
10. **`test_mcp_organ_diversity_boost`** — DiversityTracker boosts underrepresented MCPOrgan

### Integration Tests (`src/mizuki/tests/test_mcp_organ_integration.py`)

11. **`test_resource_encoding`** — JSON resource → tensor roundtrip
12. **`test_poll_loop_updates_cache`** — MCPOrganLoop polls mock server, updates organ cache
13. **`test_poll_failure_graceful`** — Failed poll doesn't crash loop, organ uses stale state
14. **`test_full_pipeline`** — Mock MCP server → organ → workspace competition → winner in ModulationState

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| MCP poll latency spikes | Could slow loop | Timeout per poll (2s), skip on timeout |
| Hash encoding loses info | Poor salience computation | Encoder network learns useful features via cycle consistency |
| Too many MCP organs flood workspace | Drowns internal organs | DiversityTracker already handles this; also set base_salience low (0.2) |
| Thread safety (poller writes, tick reads) | Stale/torn read | Tensor `.clone()` in update_cache ensures atomic-ish copy |
| MCP server disconnects | Organ goes stale | Staleness decay naturally reduces salience to 0 |

---

## Non-Goals (Out of Scope for Enhancement A)

- **Bidirectional control:** Workspace broadcast → MCP tool calls (Enhancement A is read-only; write-back is Enhancement A.2)
- **Learning MCP salience weights:** Fixed formula for now (change detection + freshness). Learned salience is a follow-up.
- **Custom encoders per MCP type:** All use the same hash→linear encoder. Type-specific encoders are a follow-up.
- **MCP server lifecycle management:** Assume servers are already connected. Startup/shutdown is existing MCPClient responsibility.

---

## Dependencies

- No new pip dependencies
- Requires: `organ.py` Organ ABC (exists), `global_workspace.py` (exists), `integration.py` register_organ (exists)
- Gated by: `CycleConfig.use_mcp_organs = True` (off by default, zero overhead when disabled)
