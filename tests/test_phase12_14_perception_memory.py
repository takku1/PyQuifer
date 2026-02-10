"""Tests for Phase 12-14: Visual/temporal/sensory binding, Active Inference, JEPA, Deliberation, Memory."""
import pytest
import torch
import math


# ── Phase 12: Visual Binding (AKOrN) ────────────────────────────────────────

class TestVisualBinding:
    def test_akorn_layer_instantiate(self):
        from pyquifer.visual_binding import AKOrNLayer
        layer = AKOrNLayer(dim=32, num_heads=4)
        assert layer is not None

    def test_akorn_layer_forward(self):
        from pyquifer.visual_binding import AKOrNLayer
        layer = AKOrNLayer(dim=32, num_heads=4)
        x = torch.randn(2, 16, 32)  # B=2, N=16 patches, D=32
        out = layer(x)
        assert out.shape == (2, 16, 32)

    def test_akorn_block_residual(self):
        from pyquifer.visual_binding import AKOrNBlock
        block = AKOrNBlock(dim=32, num_heads=4)
        x = torch.randn(2, 8, 32)
        out = block(x)
        assert out.shape == x.shape

    def test_akorn_encoder_stack(self):
        from pyquifer.visual_binding import AKOrNEncoder
        enc = AKOrNEncoder(dim=32, num_heads=4, depth=2)
        x = torch.randn(2, 8, 32)
        out = enc(x)
        assert out.shape == x.shape

    def test_oscillatory_segmenter(self):
        from pyquifer.visual_binding import OscillatorySegmenter
        seg = OscillatorySegmenter(dim=32, num_heads=4)
        x = torch.randn(2, 16, 32)
        result = seg(x)
        assert 'segments' in result
        assert 'num_segments' in result

    def test_binding_readout(self):
        from pyquifer.visual_binding import BindingReadout
        readout = BindingReadout(dim=32, max_objects=8)
        features = torch.randn(2, 16, 32)
        segment_ids = torch.randint(0, 4, (2, 16))
        result = readout(features, segment_ids)
        assert result.shape == (2, 8, 32)


# ── Phase 12: Temporal Binding ───────────────────────────────────────────────

class TestTemporalBinding:
    def test_sequence_akorn_forward(self):
        from pyquifer.temporal_binding import SequenceAKOrN
        layer = SequenceAKOrN(dim=32, num_heads=4, num_steps=3)
        x = torch.randn(2, 10, 32)  # B=2, T=10, D=32
        out = layer(x)
        assert out.shape == (2, 10, 32)

    def test_sequence_akorn_causal(self):
        from pyquifer.temporal_binding import SequenceAKOrN
        layer = SequenceAKOrN(dim=32, num_heads=4, causal=True)
        x = torch.randn(1, 8, 32)
        out = layer(x)
        assert out.shape == (1, 8, 32)

    def test_phase_grouping(self):
        from pyquifer.temporal_binding import PhaseGrouping
        grouper = PhaseGrouping(dim=32, max_groups=8)
        features = torch.randn(2, 10, 32)
        result = grouper(features)
        assert 'group_ids' in result
        assert 'num_groups' in result
        assert result['group_ids'].shape == (2, 10)

    def test_oscillatory_chunking(self):
        from pyquifer.temporal_binding import OscillatoryChunking
        chunker = OscillatoryChunking(dim=32, num_levels=3, num_steps=5)
        x = torch.randn(2, 12, 32)
        result = chunker(x)
        assert 'chunks' in result
        assert 'hierarchy' in result
        assert 'level_coherences' in result
        assert result['level_coherences'].shape == (2, 3)


# ── Phase 12: Sensory Binding ───────────────────────────────────────────────

class TestSensoryBinding:
    def test_modality_encoder(self):
        from pyquifer.sensory_binding import ModalityEncoder
        enc = ModalityEncoder(input_dim=64, oscillator_dim=32, num_oscillators=8)
        x = torch.randn(4, 64)
        result = enc(x)
        assert 'encoded' in result
        assert 'phases' in result

    def test_binding_strength(self):
        from pyquifer.sensory_binding import BindingStrength
        bs = BindingStrength()
        phases_a = torch.rand(4, 16) * 2 * math.pi
        phases_b = torch.rand(4, 16) * 2 * math.pi
        result = bs(phases_a, phases_b)
        # Returns a scalar binding strength tensor
        assert result.numel() >= 1

    def test_cross_modal_attention(self):
        from pyquifer.sensory_binding import CrossModalAttention
        cma = CrossModalAttention(dim_a=32, dim_b=32, num_heads=4)
        a = torch.randn(2, 8, 32)
        b = torch.randn(2, 8, 32)
        # Phases are (B, N_osc) — per-oscillator phases
        phases_a = torch.rand(2, 8) * 2 * math.pi
        phases_b = torch.rand(2, 8) * 2 * math.pi
        out_a, out_b = cma(a, b, phases_a, phases_b)
        assert out_a.shape == a.shape
        assert out_b.shape == b.shape

    def test_multimodal_binder(self):
        from pyquifer.sensory_binding import MultimodalBinder
        binder = MultimodalBinder(modality_dims={'vision': 64, 'audio': 48})
        inputs = {'vision': torch.randn(4, 64), 'audio': torch.randn(4, 48)}
        result = binder(inputs)
        assert 'bound_representation' in result
        assert 'total_binding' in result


# ── Phase 13: Deep Active Inference ──────────────────────────────────────────

class TestDeepActiveInference:
    def test_deep_aif_instantiate(self):
        from pyquifer.deep_active_inference import DeepAIF
        aif = DeepAIF(obs_dim=32, action_dim=4, latent_dim=16)
        assert aif is not None

    def test_latent_transition_model(self):
        from pyquifer.deep_active_inference import LatentTransitionModel
        model = LatentTransitionModel(latent_dim=16, action_dim=4)
        state = torch.randn(2, 16)
        action = torch.randn(2, 4)
        result = model(state, action)
        # forward returns tuple: (z_next, z_mean, z_logvar) or similar
        assert isinstance(result, tuple)
        assert result[0].shape[1] == 16

    def test_policy_network(self):
        from pyquifer.deep_active_inference import PolicyNetwork
        policy = PolicyNetwork(latent_dim=16, action_dim=4)
        state = torch.randn(2, 16)
        result = policy(state)
        # forward returns tuple: (mean, log_std) for continuous
        assert isinstance(result, tuple)
        assert result[0].shape == (2, 4)

    def test_multi_step_planner(self):
        from pyquifer.deep_active_inference import DeepAIF
        aif = DeepAIF(obs_dim=32, action_dim=4, latent_dim=16, horizon=3)
        obs = torch.randn(32)
        result = aif.act(obs.unsqueeze(0), use_planner=False)
        assert 'action' in result


# ── Phase 13: JEPA ──────────────────────────────────────────────────────────

class TestJEPA:
    def test_jepa_encoder(self):
        from pyquifer.jepa import JEPAEncoder
        enc = JEPAEncoder(input_dim=64, latent_dim=32)
        x = torch.randn(4, 64)
        z = enc(x)
        assert z.shape == (4, 32)

    def test_jepa_predictor(self):
        from pyquifer.jepa import JEPAPredictor
        pred = JEPAPredictor(latent_dim=32)
        z = torch.randn(4, 32)
        out = pred(z)
        assert out.shape == (4, 32)

    def test_vicreg_loss(self):
        from pyquifer.jepa import VICRegLoss
        loss_fn = VICRegLoss()
        z_pred = torch.randn(8, 32)
        z_target = torch.randn(8, 32)
        result = loss_fn(z_pred, z_target)
        assert 'loss' in result
        assert 'invariance' in result
        assert 'variance' in result

    def test_barlow_loss(self):
        from pyquifer.jepa import BarlowLoss
        loss_fn = BarlowLoss()
        z_pred = torch.randn(8, 32)
        z_target = torch.randn(8, 32)
        result = loss_fn(z_pred, z_target)
        assert 'loss' in result
        assert result['loss'].item() >= 0

    def test_action_jepa_forward(self):
        from pyquifer.jepa import ActionJEPA
        jepa = ActionJEPA(obs_dim=64, action_dim=4, latent_dim=32)
        obs_t = torch.randn(4, 64)
        action = torch.randn(4, 4)
        obs_tp1 = torch.randn(4, 64)
        result = jepa(obs_t, action, obs_tp1)
        assert 'loss' in result
        assert 'z_pred' in result

    def test_action_jepa_imagine(self):
        from pyquifer.jepa import ActionJEPA
        jepa = ActionJEPA(obs_dim=64, action_dim=4, latent_dim=32)
        obs = torch.randn(2, 64)
        actions = torch.randn(2, 5, 4)  # 5-step action plan
        traj = jepa.imagine(obs, actions)
        assert traj.shape == (2, 6, 32)  # T+1 states


# ── Phase 13: Deliberation ──────────────────────────────────────────────────

class TestDeliberation:
    def test_process_reward_model(self):
        from pyquifer.deliberation import ProcessRewardModel
        prm = ProcessRewardModel(dim=32)
        steps = torch.randn(2, 5, 32)  # B=2, 5 reasoning steps
        scores = prm(steps)
        assert scores.shape == (2, 5)

    def test_beam_search_reasoner(self):
        from pyquifer.deliberation import BeamSearchReasoner
        reasoner = BeamSearchReasoner(dim=32, beam_width=3, max_steps=4)
        # BeamSearchReasoner expects batched input (B, D)
        query = torch.randn(1, 32)
        result = reasoner(query)
        assert 'best_chain' in result
        assert 'best_scores' in result

    def test_compute_budget(self):
        from pyquifer.deliberation import ComputeBudget
        budget = ComputeBudget()
        alloc = budget.allocate(coherence=0.3)  # Low coherence → more compute
        assert alloc.num_steps > 0
        alloc_high = budget.allocate(coherence=0.9)
        # High coherence should use less compute
        assert alloc_high.num_steps <= alloc.num_steps

    def test_self_correction_loop(self):
        from pyquifer.deliberation import SelfCorrectionLoop
        loop = SelfCorrectionLoop(dim=32)
        chain = torch.randn(1, 4, 32)
        scores = torch.tensor([[0.9, 0.2, 0.8, 0.7]])  # Step 2 is bad
        result = loop(chain, scores)
        assert 'corrected_chain' in result


# ── Phase 14: CLS Memory ────────────────────────────────────────────────────

class TestCLSMemory:
    def test_hippocampal_store_recall(self):
        from pyquifer.cls_memory import HippocampalModule
        hippo = HippocampalModule(dim=32, capacity=100)
        content = torch.randn(32)
        idx = hippo.store(content)
        assert idx == 0
        result = hippo.recall(content, top_k=1)
        assert result['memories'].shape[0] == 1

    def test_neocortical_consolidation(self):
        from pyquifer.cls_memory import NeocorticalModule
        neo = NeocorticalModule(dim=32)
        traces = torch.randn(10, 32)
        result = neo.consolidate(traces)
        assert 'num_updated' in result

    def test_forgetting_curve(self):
        from pyquifer.cls_memory import ForgettingCurve
        fc = ForgettingCurve()
        ret = fc.retention(time_elapsed=0.0, access_count=0)
        assert ret == 1.0
        ret_later = fc.retention(time_elapsed=7200.0, access_count=0)
        assert ret_later < ret

    def test_importance_scorer(self):
        from pyquifer.cls_memory import ImportanceScorer
        scorer = ImportanceScorer()
        score = scorer.score(emotional_salience=0.8, novelty=0.6, utility=0.5, recency=0.3)
        assert 0 <= score <= 1

    def test_memory_interference(self):
        from pyquifer.cls_memory import MemoryInterference
        mi = MemoryInterference(dim=32)
        new = torch.randn(32)
        existing = torch.randn(5, 32)
        result = mi.check(new, existing)
        assert 'interference_risk' in result


# ── Phase 14: Temporal Knowledge Graph ───────────────────────────────────────

class TestTemporalGraph:
    def test_graph_add_node_edge(self):
        from pyquifer.temporal_graph import TemporalKnowledgeGraph, TemporalNode, TemporalEdge
        g = TemporalKnowledgeGraph()
        g.add_node(TemporalNode("A"))
        g.add_node(TemporalNode("B"))
        g.add_edge(TemporalEdge("A", "B", "knows", t_start=0, t_end=10))
        assert g.num_nodes == 2
        assert g.num_edges == 1

    def test_temporal_query(self):
        from pyquifer.temporal_graph import TemporalKnowledgeGraph, TemporalNode, TemporalEdge
        g = TemporalKnowledgeGraph()
        g.add_node(TemporalNode("A"))
        g.add_node(TemporalNode("B"))
        g.add_edge(TemporalEdge("A", "B", "knows", t_start=0, t_end=5))
        neighbors = g.get_neighbors("A", at_time=3.0)
        assert len(neighbors) == 1
        neighbors_late = g.get_neighbors("A", at_time=10.0)
        assert len(neighbors_late) == 0

    def test_graph_snapshot(self):
        from pyquifer.temporal_graph import TemporalKnowledgeGraph, TemporalNode, TemporalEdge
        g = TemporalKnowledgeGraph()
        g.add_node(TemporalNode("X"))
        g.add_node(TemporalNode("Y"))
        g.add_edge(TemporalEdge("X", "Y", "causes", t_start=1, t_end=5))
        snap = g.snapshot(at_time=3.0)
        assert snap['num_edges'] == 1

    def test_event_timeline(self):
        from pyquifer.temporal_graph import EventTimeline, Event
        tl = EventTimeline()
        tl.add_event(Event("e1", "first", timestamp=1.0))
        tl.add_event(Event("e2", "second", timestamp=2.0))
        before = tl.get_before(1.5)
        assert len(before) == 1
        assert before[0].event_id == "e1"

    def test_temporal_reasoner(self):
        from pyquifer.temporal_graph import TemporalReasoner
        reasoner = TemporalReasoner(dim=32, num_relations=8)
        subject = torch.randn(2, 32)
        rel = torch.tensor([0, 1])
        time = torch.tensor([1.0, 2.0])
        scores = reasoner.score_triple(subject, rel, time)
        assert scores.shape == (2,)


# ── Phase 14: Gated Memory ──────────────────────────────────────────────────

class TestGatedMemory:
    def test_nmda_gate(self):
        from pyquifer.gated_memory import NMDAGate
        gate = NMDAGate(dim=32)
        content = torch.randn(4, 32)
        phase = torch.tensor([math.pi] * 4)  # At theta peak
        gate_val, gated = gate(content, phase)
        assert gate_val.shape == (4, 32)
        assert gated.shape == (4, 32)

    def test_memory_bank_read_write(self):
        from pyquifer.gated_memory import DifferentiableMemoryBank
        bank = DifferentiableMemoryBank(num_slots=64, slot_dim=32)
        content = torch.randn(32)
        phase = torch.tensor(math.pi)  # Theta peak
        write_result = bank.write(content, phase)
        assert 'gate_value' in write_result
        read_result = bank.read(content)
        assert read_result['content'].shape == (32,)

    def test_consolidation_loop(self):
        from pyquifer.gated_memory import MemoryConsolidationLoop
        loop = MemoryConsolidationLoop(dim=32, num_replay_steps=3)
        traces = torch.randn(5, 32)
        result = loop.consolidate(traces)
        assert result['num_replayed'] == 3

    def test_memory_bank_reset(self):
        from pyquifer.gated_memory import DifferentiableMemoryBank
        bank = DifferentiableMemoryBank(num_slots=32, slot_dim=16)
        bank.write(torch.randn(16), torch.tensor(1.0))
        bank.reset()
        assert bank.usage.sum().item() == 0.0
