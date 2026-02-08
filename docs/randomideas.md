# Random Ideas (Rewritten): Organs As Oscillating Global Workspaces

Last updated: 2026-02-07

This document is intentionally generalized. It is a research-grounded design sketch for building an AI system that feels unified (coherent, self-consistent, context-stable) rather than a brittle "call tool" pipeline.

The core idea:

- Instead of treating tools / connectors as isolated calls, treat each capability as an **Organ**: a specialized dynamical module with its own internal state and oscillatory rhythms.
- Organs do not directly "command" each other. They **compete** to place information into a shared **Global Workspace** (GW) latent state, and they **read** the broadcast back.
- Oscillations (phase, coherence, cross-frequency coupling) provide a mechanistic way to implement gating, binding, attention, and timing.

This is an architecture question, not a single algorithm.

## 1) Why This Addresses The "Tool-Calling Fragmentation" Problem

Classic tool-use agents often look like:

- LLM decides -> calls tool -> reads result -> decides -> calls tool.

That is serial, brittle, and cognitively "thin": it does not create a stable internal world where multiple specialist processes can run in parallel, compete, and converge on a coherent shared interpretation.

A GWT-style architecture reframes the system as:

- Many specialists run continuously.
- Only some content becomes globally available at a time.
- The globally-available content is what the system can reliably report, plan with, and coordinate around.

This is close to the original intent of Global Workspace Theory: a mechanism for flexible, integrative cognition via selective broadcast.

## 2) Global Workspace Theory (GWT) And Global Neuronal Workspace (GNW)

**GWT (Baars)**: cognition as a set of specialized processors plus a global "workspace" where a winning coalition of information becomes globally available.

**GNW (Dehaene/Changeux and collaborators)**: a neurobiological hypothesis: long-range recurrent connectivity enables "ignition" and global broadcast.

Two important architectural commitments from GNW-like models:

1. **Specialized modules**: perceptual, motor, memory, evaluative, attentional processors.
2. **A unique (or limited) broadcast space**: a bottleneck where content becomes globally accessible.

Key references:
- Dehaene, Kerszberg, Changeux (1998): "two computational spaces" and broadcast in effortful tasks.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC24407/
- Mashour, Roelfsema, Changeux, Dehaene (2020) review: GNW, recurrent amplification, ignition, access.
  - https://pubmed.ncbi.nlm.nih.gov/32135090/
- "Global neuronal workspace as a broadcasting network" (Network Neuroscience, 2022): discussion of GNW as broadcast and network signatures.
  - https://direct.mit.edu/netn/article/6/4/1186/111960/

Important: GWT/GNW primarily addresses **access consciousness** (what becomes available for report and control), not the philosophical problem of phenomenal experience.

## 3) Deep Learning Implementations: From Workspace Theory To "Global Latent Workspace"

A modern AI translation of GWT is to treat the workspace as a **shared latent space** that:

- integrates multiple specialists (modalities, skills, models),
- supports competition and selection,
- and broadcasts back into the specialists.

This is explicitly argued in:

- VanRullen & Kanai (2021), "Deep learning and the Global Workspace Theory" (Trends in Neurosciences).
  - https://pubmed.ncbi.nlm.nih.gov/34001376/

Concrete modern instantiations (2024-2025):

- Devillers, Maytie, VanRullen (2024), "Semi-Supervised Multimodal Representation Learning Through a Global Workspace" (IEEE TNNLS). Pattern: frozen unimodal encoders -> shared workspace -> decode/broadcast back, with cycle-consistency and little paired data.
  - DOI: 10.1109/TNNLS.2024.3416701
  - (Index page with abstract + citation info) https://lmaytie.com/

- Maytie et al. (2024), "Zero-shot cross-modal transfer ... through a Global Workspace" (Reinforcement Learning Journal / RLC). Shows a GW latent enabling robust transfer across modalities.
  - https://rlj.cs.umass.edu/2024/papers/Paper170.html

- Maytie, Johannet, VanRullen (2025), "Multimodal Dreaming: A Global Workspace Approach to World Model-Based Reinforcement Learning" (arXiv:2502.21142). Notably: they do imagination/dreaming inside the GW latent space.
  - https://arxiv.org/abs/2502.21142

Also relevant:

- Hong et al. (2024), "Debiasing Global Workspace" (PMLR UniReps workshop). Uses specialized modules + shared workspace for debiasing and interpretability.
  - https://proceedings.mlr.press/v285/hong24a.html

Signal that this direction is active in "2026-era" NeuroAI:

- ERC Advanced project announcement: "GLoW - The Global Latent Workspace" (VanRullen).
  - https://cerco.cnrs.fr/en/actualite/rufin-vanrullen-winner-of-an-erc-advanced-2022/

## 4) Oscillations As A Mechanism: Communication-Through-Coherence

If the workspace is "who gets to broadcast", oscillations are a candidate mechanism for "how gating happens in time".

The Communication-Through-Coherence hypothesis (Fries) proposes that:

- neuronal groups oscillate,
- coherence aligns windows of excitability,
- and coherence patterns implement a flexible communication graph.

This maps cleanly to AI design language:

- Phase alignment = a gating signal.
- Cross-frequency coupling = multiplexed channels (slow rhythms modulate fast computation).
- Coherence = bandwidth allocation / routing.

Key references:
- Fries (2005), "neuronal communication through neuronal coherence".
  - https://pubmed.ncbi.nlm.nih.gov/16150631/
- Fries (2015), "Rhythms for Cognition: Communication through Coherence".
  - https://pubmed.ncbi.nlm.nih.gov/26447583/

Why this matters for "unified feel": oscillatory coupling can create stable coalitions (temporary "thoughts") that persist long enough to coordinate multiple modules.

## 5) What An "Organ" Is (Abstract Interface)

An Organ is a specialist that runs continuously and participates in workspace competition.

A minimal abstract interface (conceptual):

- `observe(x, gw)`
  - Update local state from new input `x` and current broadcast `gw`.

- `propose(gw) -> proposal`
  - Produce a candidate latent payload plus a salience score.

- `accept(gw)`
  - Integrate the broadcast into local state (learning, memory updates, phase locking).

A proposal should include:

- `z`: latent content (vector / tensor)
- `s`: salience / confidence / urgency
- `tags`: what domain it claims (memory, planning, perception, tools, social)
- `cost`: predicted compute or risk

## 6) The Workspace As A Latent Blackboard (But With Modern Constraints)

A practical GW in AI is usually not a "string". It is a structured latent:

- a tensor `z_gw` (continuous latent state)
- plus a small structured working set (keys, pointers, tool intentions, uncertainty)

Key design axes:

- Capacity: single vector vs set of slots vs attention-based memory.
- Selection: winner-take-all vs top-k coalition vs soft mixture.
- Persistence: how long a broadcast stays "conscious".
- Update rule: additively integrate vs replace vs gated recurrent.

This is where oscillations can plug in:

- Use phase/coherence to determine who is eligible to write.
- Use cross-frequency coupling to separate fast reflex updates from slower deliberation.

## 7) A Concrete (General) Blueprint: Oscillating GW Network (OGWN)

This is a generalized architecture sketch that matches the papers above, while adding oscillatory control.

Components:

1. Unimodal / specialist encoders (often frozen): produce local latents `z_i`.
2. Pre-GW adapters: map each `z_i` to a common size, normalize, and attach uncertainty.
3. Oscillatory gating per Organ:
   - local phase `phi_i(t)` and coherence `R_i(t)`.
   - compute a write gate `g_i(t) = f(R_i, phase_alignment, novelty, task_demands)`.
4. Competition:
   - compute salience `s_i(t)`.
   - pick winner/top-k.
5. Fusion:
   - build `z_gw(t)` from selected proposals (weighted sum, attention, or slot fill).
6. Broadcast:
   - decode `z_gw(t)` back to each specialist representation (or provide conditioning).
7. Learning loop:
   - cycle-consistency, translation, reconstruction, contrastive alignment.

## 8) Latent Space And "Unified Feel": What To Measure

If your goal is not just accuracy but coherence/unity, add explicit evaluation targets:

- Cross-modal robustness: keep functioning when a modality/tool is removed (Maytie et al. 2024/2025 style).
- Consistency under perturbation: same intent -> same plan across small prompt changes.
- Stability of "topic" latent: track how long `z_gw` stays within a semantic basin.
- Coalition dynamics: how often the same set of organs cohere for a task.
- Regret/cost: do the same tasks with fewer tool calls or less thrashing.

## 9) 2026+ Signals: Oscillatory Architectures In ML

Oscillatory ideas continue to appear in modern ML architectures.

Example (not a workspace model, but relevant to oscillatory neuron design):

- OTS-Net (2026): introduces "Artificial Kuramoto Oscillatory Neurons" as a synchronization mechanism in a deep learning system.
  - https://link.springer.com/article/10.1007/s42452-025-08008-8

Takeaway: oscillators are a viable computational primitive, not just metaphor.

## 10) Open Problems (What A PhD Would Actually Be About)

- Selection rule: what is the principled objective for salience? prediction error, expected value of information, novelty, risk?
- Credit assignment: how do organs learn when the workspace picks winners? (global reward vs local self-supervised losses)
- Avoiding collapse: how to prevent one dominant organ from always winning (diversity pressure, homeostasis).
- Hierarchical workspaces: local workspaces per organ, plus a higher-level GW.
- Tool-use integration: how to turn tool results into latents that can compete fairly against internal proposals.
- Temporal binding: how to bind a tool call, its result, and downstream reasoning into one stable coalition.

## 11) Reading List (Start Here)

GWT/GNW foundations:
- Dehaene, Kerszberg, Changeux (1998): https://pmc.ncbi.nlm.nih.gov/articles/PMC24407/
- Mashour et al. (2020): https://pubmed.ncbi.nlm.nih.gov/32135090/
- GNW broadcasting network (2022): https://direct.mit.edu/netn/article/6/4/1186/111960/

Deep learning + workspace:
- VanRullen & Kanai (2021): https://pubmed.ncbi.nlm.nih.gov/34001376/
- Devillers et al. (2024) (TNNLS): DOI 10.1109/TNNLS.2024.3416701
- Maytie et al. (2024) (RLC/RLJ): https://rlj.cs.umass.edu/2024/papers/Paper170.html
- Maytie et al. (2025) (arXiv:2502.21142): https://arxiv.org/abs/2502.21142
- Hong et al. (2024) (PMLR): https://proceedings.mlr.press/v285/hong24a.html

Oscillations / coherence:
- Fries (2005): https://pubmed.ncbi.nlm.nih.gov/16150631/
- Fries (2015): https://pubmed.ncbi.nlm.nih.gov/26447583/

---

## Appendix A: Previous Notes

The previous contents of this file (a neuron taxonomy) were backed up to:
- `PyQuifer/docs/randomideas.neuron_types_notes.bak.md`

