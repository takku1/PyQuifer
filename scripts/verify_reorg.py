"""Verify all classes, methods, and functions survived the reorganization."""
import ast
import os
import sys

copy_dir = 'src/pyquifer - Copy'
new_dir = 'src/pyquifer'

file_map = {
    'integration.py': ['runtime/config.py', 'runtime/tick_result.py', 'runtime/criticality_feedback.py', 'runtime/cycle.py'],
    'oscillators.py': ['dynamics/oscillators/kuramoto.py', 'dynamics/oscillators/stuart_landau.py', 'dynamics/oscillators/mean_field.py'],
    'global_workspace.py': ['workspace/workspace.py', 'workspace/competition.py', 'workspace/broadcast.py', 'workspace/ensemble.py'],
    'criticality.py': ['dynamics/criticality/monitors.py', 'dynamics/criticality/controllers.py'],
    'core.py': ['api/legacy.py'],
    'bridge.py': ['api/bridge.py'],
    'models.py': ['memory/generative_world_model.py'],
    'world_model.py': ['memory/latent_world_model.py'],
    'neuro_diagnostics.py': ['diagnostics/neuroscience.py'],
    'kindchenschema.py': ['embodiment/safety.py'],
    'self_model.py': ['identity/self_model.py'],
    'strange_attractor.py': ['identity/strange_attractor.py'],
    'consciousness.py': ['identity/consciousness.py'],
    'neural_darwinism.py': ['identity/neural_darwinism.py'],
    'adapter_manager.py': ['identity/adapter_manager.py'],
    'hypernetwork.py': ['identity/hypernetwork.py'],
    'quantum_cognition.py': ['experimental/quantum.py'],
    'hyperdimensional.py': ['experimental/hyperdimensional.py'],
    'hyperbolic.py': ['experimental/hyperbolic.py'],
    'reservoir.py': ['experimental/reservoir.py'],
    'mcp_organ.py': ['workspace/organ_mcp.py'],
    'organ.py': ['workspace/organ_base.py'],
    'somatic.py': ['embodiment/somatic.py'],
    'morphological.py': ['embodiment/morphology.py'],
    'ecology.py': ['embodiment/ecology.py'],
    'social.py': ['embodiment/social.py'],
    'developmental.py': ['embodiment/developmental.py'],
    'voice_dynamics.py': ['embodiment/voice.py'],
    'motivation.py': ['cognition/control/motivation.py'],
    'metacognitive.py': ['cognition/control/metacognitive.py'],
    'deliberation.py': ['cognition/control/deliberation.py'],
    'volatility_filter.py': ['cognition/control/volatility.py'],
    'hierarchical_predictive.py': ['cognition/predictive/hierarchical.py'],
    'active_inference.py': ['cognition/predictive/active_inference.py'],
    'deep_active_inference.py': ['cognition/predictive/deep_active_inference.py'],
    'jepa.py': ['cognition/predictive/jepa.py'],
    'causal_flow.py': ['cognition/reasoning/causal_flow.py'],
    'causal_reasoning.py': ['cognition/reasoning/causal_reasoning.py'],
    'graph_reasoning.py': ['cognition/reasoning/graph_reasoning.py'],
    'temporal_graph.py': ['cognition/reasoning/temporal_graph.py'],
    'phase_attention.py': ['cognition/attention/phase_attention.py'],
    'precision_weighting.py': ['cognition/attention/precision_weighting.py'],
    'visual_binding.py': ['cognition/binding/visual.py'],
    'temporal_binding.py': ['cognition/binding/temporal.py'],
    'sensory_binding.py': ['cognition/binding/sensory.py'],
    'selective_ssm.py': ['cognition/routing/ssm.py'],
    'oscillatory_moe.py': ['cognition/routing/moe.py'],
    'flash_rnn.py': ['cognition/routing/flash_rnn.py'],
    'learning.py': ['learning/synaptic.py'],
    'continual_learning.py': ['learning/continual.py'],
    'memory_consolidation.py': ['learning/consolidation.py'],
    'equilibrium_propagation.py': ['learning/equilibrium_prop.py'],
    'prospective_config.py': ['learning/prospective.py'],
    'short_term_plasticity.py': ['learning/stp.py'],
    'spiking.py': ['dynamics/spiking/neurons.py'],
    'advanced_spiking.py': ['dynamics/spiking/advanced.py'],
    'energy_spiking.py': ['dynamics/spiking/energy.py'],
    'stochastic_resonance.py': ['dynamics/stochastic.py'],
    'metastability.py': ['dynamics/metastability.py'],
    'neuromodulation.py': ['dynamics/neuromodulation.py'],
    'linoss.py': ['dynamics/continuous/linoss.py'],
    'liquid_networks.py': ['dynamics/continuous/liquid.py'],
    'neural_mass.py': ['dynamics/continuous/neural_mass.py'],
    'ode_solvers.py': ['dynamics/continuous/ode_solvers.py'],
    'thermodynamic.py': ['dynamics/continuous/thermodynamic.py'],
    'complex_oscillators.py': ['dynamics/oscillators/complex.py'],
    'spherical.py': ['dynamics/oscillators/spherical.py'],
    'frequency_bank.py': ['dynamics/oscillators/frequency_bank.py'],
    'multiplexing.py': ['dynamics/oscillators/coupling.py'],
    'gated_memory.py': ['memory/gated_memory.py'],
    'cls_memory.py': ['memory/cls.py'],
    'diffusion.py': ['memory/diffusion.py'],
    'perturbation.py': ['memory/perturbation.py'],
    'potentials.py': ['memory/potentials.py'],
    'appraisal.py': ['cognition/control/appraisal.py'],
    'iit_metrics.py': ['identity/iit_metrics.py'],
    'dendritic.py': ['learning/dendritic.py'],
    'dendritic_learning.py': ['learning/dendritic_learning.py'],
    'fhrr.py': ['experimental/fhrr.py'],
    'basal_ganglia.py': ['dynamics/basal_ganglia.py'],
    'phase_lock_bus.py': ['dynamics/phase_lock_bus.py'],
}


def get_class_methods(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return {}, []
    classes = {}
    funcs = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = sorted(
                n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            classes[node.name] = methods
        elif isinstance(node, ast.FunctionDef):
            funcs.append(node.name)
    return classes, funcs


missing_classes = []
missing_methods = []
missing_funcs = []
total_classes = 0
total_methods = 0
total_funcs = 0

for old_name, new_paths in sorted(file_map.items()):
    old_path = os.path.join(copy_dir, old_name)
    if not os.path.exists(old_path):
        continue

    old_classes, old_funcs = get_class_methods(old_path)

    # Collect from new files
    new_classes = {}
    new_funcs = []
    for np in new_paths:
        full = os.path.join(new_dir, np)
        if os.path.exists(full):
            nc, nf = get_class_methods(full)
            new_classes.update(nc)
            new_funcs.extend(nf)

    # Compare classes
    for cls_name, old_methods in old_classes.items():
        total_classes += 1
        if cls_name not in new_classes:
            missing_classes.append(f"{cls_name} (from {old_name})")
            continue
        new_methods = new_classes[cls_name]
        for m in old_methods:
            total_methods += 1
            if m not in new_methods:
                missing_methods.append(f"{cls_name}.{m} (from {old_name})")

    # Compare top-level functions
    for fn in old_funcs:
        total_funcs += 1
        if fn not in new_funcs:
            missing_funcs.append(f"{fn} (from {old_name})")

print(f"Classes checked: {total_classes}")
print(f"Methods checked: {total_methods}")
print(f"Top-level functions checked: {total_funcs}")
print()
if missing_classes:
    print(f"=== MISSING CLASSES ({len(missing_classes)}) ===")
    for m in missing_classes:
        print(f"  {m}")
if missing_methods:
    print(f"=== MISSING METHODS ({len(missing_methods)}) ===")
    for m in missing_methods:
        print(f"  {m}")
if missing_funcs:
    print(f"=== MISSING FUNCTIONS ({len(missing_funcs)}) ===")
    for m in missing_funcs:
        print(f"  {m}")
if not missing_classes and not missing_methods and not missing_funcs:
    print("ALL CLASSES, METHODS, AND FUNCTIONS VERIFIED PRESENT!")
