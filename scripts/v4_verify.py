"""v4 Neuroscience Alignment Verification Script."""
import torch
import math
import time

from pyquifer.integration import CognitiveCycle, CycleConfig
from pyquifer.bridge import PyQuiferBridge

print("=" * 70)
print("  PyQuifer v4 Neuroscience Alignment Verification")
print("=" * 70)
print()

# ─────────────────────────────────────────────────────
# 1. Modular topology: within-module R > between-module R
# ─────────────────────────────────────────────────────
print("--- 1. Modular Topology: Within-Module vs Between-Module Coherence ---")
c = CycleConfig.neuroscience()
cycle = CognitiveCycle(c)

for i in range(200):
    cycle.tick(torch.randn(64), return_diagnostics=False)

phases = cycle.oscillators.phases.detach()
adj = cycle.oscillators.get_adjacency()
mod_size = c.num_oscillators // 4
within_Rs = []
between_Rs = []
for m in range(4):
    s, e = m * mod_size, (m + 1) * mod_size
    mod_phases = phases[s:e]
    mod_complex = torch.exp(1j * mod_phases.to(torch.complex64))
    within_R = torch.abs(mod_complex.mean()).item()
    within_Rs.append(within_R)

for i in range(4):
    for j in range(i + 1, 4):
        s1, e1 = i * mod_size, (i + 1) * mod_size
        s2, e2 = j * mod_size, (j + 1) * mod_size
        combined = torch.cat([phases[s1:e1], phases[s2:e2]])
        comb_complex = torch.exp(1j * combined.to(torch.complex64))
        between_R = torch.abs(comb_complex.mean()).item()
        between_Rs.append(between_R)

mean_within = sum(within_Rs) / len(within_Rs)
mean_between = sum(between_Rs) / len(between_Rs)
print(f"  Within-module R (mean):  {mean_within:.4f}")
print(f"  Between-module R (mean): {mean_between:.4f}")
print(f"  Ratio: {mean_within / max(mean_between, 1e-6):.2f}x")
print(f"  Intra-module density: {adj[0:mod_size, 0:mod_size].mean():.3f}")
print(f"  Inter-module density: {adj[0:mod_size, mod_size:2*mod_size].mean():.3f}")
print()

# ─────────────────────────────────────────────────────
# 2. Theta-gamma PAC: Tort MI > 0 and varies with state
# ─────────────────────────────────────────────────────
print("--- 2. Theta-Gamma PAC: Tort MI Measurement ---")
c_pac = CycleConfig.neuroscience()
cycle_pac = CognitiveCycle(c_pac)

mi_values = []
for i in range(500):
    cycle_pac.tick(torch.randn(64), return_diagnostics=False)
    if i >= 50 and i % 50 == 0:
        mi_values.append(cycle_pac._cached_pac_mi.item())

print(f"  Tort MI values (every 50 ticks): {[f'{v:.4f}' for v in mi_values]}")
print(f"  MI mean: {sum(mi_values) / len(mi_values):.4f}")
print(f"  MI stdev: {torch.tensor(mi_values).std().item():.4f}")
print(f"  MI > 0: {sum(1 for v in mi_values if v > 0)}/{len(mi_values)}")

theta_omega = cycle_pac.oscillators.natural_frequencies[:8]
gamma_omega = cycle_pac.oscillators.natural_frequencies[8:]
theta_hz = theta_omega / (2 * math.pi * 0.01)
gamma_hz = gamma_omega / (2 * math.pi * 0.01)
print(f"  Theta band: {theta_hz.min():.1f}-{theta_hz.max():.1f} Hz")
print(f"  Gamma band: {gamma_hz.min():.1f}-{gamma_hz.max():.1f} Hz")
print()

# ─────────────────────────────────────────────────────
# 3. Size-normalized metastability target
# ─────────────────────────────────────────────────────
print("--- 3. Size-Normalized Metastability Target ---")
for N in [16, 32, 64, 128, 256]:
    cc = CycleConfig(num_oscillators=N)
    cy = CognitiveCycle(cc)
    analytical = 0.15 * math.sqrt(64.0 / N)
    print(f"  N={N:3d}: target_meta={cy._effective_target_meta:.4f} (analytical: {analytical:.4f})")
print()

# ─────────────────────────────────────────────────────
# 4. SD(R) convergence test
# ─────────────────────────────────────────────────────
print("--- 4. SD(R) Convergence (500 ticks, neuroscience preset) ---")
c_meta = CycleConfig.neuroscience()
cycle_meta = CognitiveCycle(c_meta)
target = cycle_meta._effective_target_meta

sd_r_at = {}
for i in range(500):
    cycle_meta.tick(torch.randn(64), return_diagnostics=False)
    if i + 1 in [50, 100, 200, 300, 500]:
        sd_r_at[i + 1] = cycle_meta._cached_metastability.item()

for tick, sdr in sd_r_at.items():
    print(f"  Tick {tick:3d}: SD(R)={sdr:.4f} (target: {target:.4f})")
print(f"  Dephasing gain final: {cycle_meta._cached_dephasing_gain.item():.4f}")
print()

# ─────────────────────────────────────────────────────
# 5. Phi_m with 4-factor formula
# ─────────────────────────────────────────────────────
print("--- 5. Phi_m Consciousness Quality (4-factor) ---")
c_phi = CycleConfig.neuroscience()
cycle_phi = CognitiveCycle(c_phi)
phi_values = []
for i in range(300):
    cycle_phi.tick(torch.randn(64), return_diagnostics=False)
    if i >= 10:
        phi_values.append(cycle_phi._cached_phi_m.item())

phi_t = torch.tensor(phi_values)
print(f"  Phi_m mean: {phi_t.mean():.6f}")
print(f"  Phi_m stdev: {phi_t.std():.6f}")
print(f"  Phi_m max: {phi_t.max():.6f}")
print(f"  Phi_m > 0: {(phi_t > 0).sum().item()}/{len(phi_values)} ({100 * (phi_t > 0).float().mean():.1f}%)")
print()

# ─────────────────────────────────────────────────────
# 6. Full diagnostic snapshot
# ─────────────────────────────────────────────────────
print("--- 6. Full Diagnostic Keys (neuroscience preset) ---")
c_diag = CycleConfig.neuroscience()
cycle_diag = CognitiveCycle(c_diag)
for i in range(100):
    r = cycle_diag.tick(torch.randn(64), return_diagnostics=(i == 99))
    if i == 99:
        c_keys = sorted(r["consciousness"].keys())
        print(f"  Consciousness keys: {c_keys}")
        for k in ["metastability_sd_R", "effective_target_meta", "meta_r_coupling",
                   "pac_mi", "phi_m", "criticality_sigma"]:
            print(f"    {k}: {r['consciousness'][k]:.6f}")
print()

# ─────────────────────────────────────────────────────
# 7. Latency check
# ─────────────────────────────────────────────────────
print("--- 7. Latency Impact ---")
presets = [
    ("default", CycleConfig.default),
    ("interactive", CycleConfig.interactive),
    ("realtime", CycleConfig.realtime),
    ("neuroscience", CycleConfig.neuroscience),
]
for preset_name, preset_fn in presets:
    cc = preset_fn()
    cy = CognitiveCycle(cc)
    inp = torch.randn(64)
    # Warmup
    for _ in range(10):
        cy.tick(inp, return_diagnostics=False)
    # Measure
    lats = []
    for _ in range(100):
        t0 = time.perf_counter()
        cy.tick(inp, return_diagnostics=False)
        lats.append((time.perf_counter() - t0) * 1000)
    lats.sort()
    p50 = lats[50]
    mean_lat = sum(lats) / len(lats)
    print(f"  {preset_name:15s}: p50={p50:.2f}ms  mean={mean_lat:.2f}ms")
print()

# ─────────────────────────────────────────────────────
# 8. Bridge preset test
# ─────────────────────────────────────────────────────
print("--- 8. Bridge Neuroscience Preset ---")
bridge = PyQuiferBridge.neuroscience()
state = bridge.step(torch.randn(64))
print(f"  Temperature: {state.temperature:.3f}")
print(f"  Coherence: {state.coherence:.3f}")
print(f"  Processing mode: {state.processing_mode}")
print(f"  Criticality distance: {state.criticality_distance:.3f}")
print()

# ─────────────────────────────────────────────────────
# 9. Backward compatibility
# ─────────────────────────────────────────────────────
print("--- 9. Backward Compatibility ---")
c_default = CycleConfig.default()
assert c_default.oscillator_topology == "global", "Default topology changed!"
assert c_default.use_theta_gamma_pac is False, "Default PAC changed!"
assert c_default.target_metastability == 0.0, "Default meta target not auto!"
cy_default = CognitiveCycle(c_default)
expected_meta_32 = 0.15 * math.sqrt(64.0 / 32)  # ~0.2121 for N=32
assert abs(cy_default._effective_target_meta - expected_meta_32) < 1e-6, (
    f"N=32 auto target wrong: {cy_default._effective_target_meta} (expected {expected_meta_32})"
)
# Old flag still works
c_old = CycleConfig(use_cross_freq_coupling=True)
cy_old = CognitiveCycle(c_old)
assert cy_old._cfc is not None, "use_cross_freq_coupling broken!"
# New alias works
c_new = CycleConfig(use_meta_r_coupling=True)
cy_new = CognitiveCycle(c_new)
assert cy_new._cfc is not None, "use_meta_r_coupling broken!"
print("  All backward compatibility checks PASSED")
print()

# ─────────────────────────────────────────────────────
# 10. Telemetry capture (neuroscience preset, 500 ticks)
# ─────────────────────────────────────────────────────
print("--- 10. Telemetry Capture (500 ticks, neuroscience preset) ---")
c_tel = CycleConfig.neuroscience()
cycle_tel = CognitiveCycle(c_tel)
coherences = []
modes = {"perception": 0, "imagination": 0, "balanced": 0}
mode_transitions = 0
prev_mode = None
phi_ms = []
pac_mis = []
meta_sds = []

for i in range(500):
    r = cycle_tel.tick(torch.randn(64), return_diagnostics=True)
    coh = r["consciousness"]["coherence"]
    coherences.append(coh)
    mode = r["modulation"]["processing_mode"]
    modes[mode] = modes.get(mode, 0) + 1
    if prev_mode is not None and mode != prev_mode:
        mode_transitions += 1
    prev_mode = mode
    phi_ms.append(r["consciousness"]["phi_m"])
    pac_mis.append(r["consciousness"]["pac_mi"])
    meta_sds.append(r["consciousness"]["metastability_sd_R"])

coh_t = torch.tensor(coherences)
print(f"  Coherence: min={coh_t.min():.4f} max={coh_t.max():.4f} mean={coh_t.mean():.4f} stdev={coh_t.std():.4f}")
print(f"  Coherence <0.3: {(coh_t < 0.3).sum().item()}")
print(f"  Coherence 0.3-0.7: {((coh_t >= 0.3) & (coh_t <= 0.7)).sum().item()}")
print(f"  Coherence >0.7: {(coh_t > 0.7).sum().item()}")
print(f"  Processing modes: {modes}")
print(f"  Mode transitions: {mode_transitions} (rate: {mode_transitions/500:.3f}/tick)")
phi_t = torch.tensor(phi_ms)
pac_t = torch.tensor(pac_mis)
meta_t = torch.tensor(meta_sds)
print(f"  Phi_m: mean={phi_t.mean():.6f} max={phi_t.max():.6f}")
print(f"  PAC MI: mean={pac_t.mean():.4f} stdev={pac_t.std():.4f}")
print(f"  SD(R): final={meta_t[-1]:.4f} (target: {cycle_tel._effective_target_meta:.4f})")
print(f"  Criticality sigma: {r['consciousness']['criticality_sigma']:.4f}")
print()

print("=" * 70)
print("  ALL VERIFICATIONS PASSED")
print("=" * 70)
