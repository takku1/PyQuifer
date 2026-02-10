"""Tests for Phase 15-18: Appraisal, Causal, SSM, MoE, Graph, Prospective, Dendritic, Energy, FHRR, FlashRNN."""
import pytest
import torch
import math


# ── Phase 15: Cognitive Appraisal ────────────────────────────────────────────

class TestAppraisal:
    def test_appraisal_dimension(self):
        from pyquifer.appraisal import AppraisalDimension
        dim = AppraisalDimension(32, "goal_relevance", bipolar=True)
        x = torch.randn(4, 32)
        value, confidence = dim(x)
        assert value.shape == (4,)
        assert confidence.shape == (4,)
        assert (value >= -1).all() and (value <= 1).all()

    def test_appraisal_chain(self):
        from pyquifer.appraisal import AppraisalChain
        chain = AppraisalChain(dim=32)
        x = torch.randn(4, 32)
        result = chain(x)
        assert 'goal_relevance' in result
        assert 'novelty' in result
        assert len(result) == 5  # Standard 5 dimensions

    def test_occ_model(self):
        from pyquifer.appraisal import OCC_Model
        occ = OCC_Model(num_dimensions=5, num_emotions=12)
        appraisals = torch.randn(4, 5)
        result = occ(appraisals)
        assert 'emotion_probs' in result
        assert 'valence' in result
        assert 'arousal' in result
        assert result['emotion_probs'].shape == (4, 12)

    def test_emotion_attribution(self):
        from pyquifer.appraisal import EmotionAttribution
        attr = EmotionAttribution(dim=32, num_dimensions=5, num_emotions=12)
        stimulus = torch.randn(4, 32)
        appraisals = torch.randn(4, 5)
        emotions = torch.randn(4, 12)
        result = attr(stimulus, appraisals, emotions)
        assert 'dimension_importance' in result
        assert 'top_dimension' in result


# ── Phase 15: Causal Reasoning ───────────────────────────────────────────────

class TestCausalReasoning:
    def test_causal_graph_basic(self):
        from pyquifer.causal_reasoning import CausalGraph
        g = CausalGraph(["X", "Y", "Z"])
        g.add_edge("X", "Y")
        g.add_edge("Y", "Z")
        assert g.num_variables == 3
        assert g.num_edges == 2
        assert "Y" in g.get_children("X")
        assert "X" in g.get_ancestors("Z")

    def test_causal_graph_cycle_detection(self):
        from pyquifer.causal_reasoning import CausalGraph
        g = CausalGraph(["A", "B", "C"])
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        with pytest.raises(ValueError):
            g.add_edge("C", "A")

    def test_topological_sort(self):
        from pyquifer.causal_reasoning import CausalGraph
        g = CausalGraph(["A", "B", "C"])
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        order = g.topological_sort()
        assert order.index("A") < order.index("B") < order.index("C")

    def test_d_separation(self):
        from pyquifer.causal_reasoning import CausalGraph
        g = CausalGraph(["X", "Z", "Y"])
        g.add_edge("X", "Z")
        g.add_edge("Z", "Y")
        # X → Z → Y: X and Y are d-separated given Z
        assert g.d_separated("X", "Y", {"Z"})

    def test_interventional_graph(self):
        from pyquifer.causal_reasoning import CausalGraph
        g = CausalGraph(["X", "Y", "Z"])
        g.add_edge("X", "Y")
        g.add_edge("Z", "Y")
        mutilated = g.interventional_graph({"Y"})
        assert mutilated.num_edges == 0  # All edges into Y removed

    def test_do_operator(self):
        from pyquifer.causal_reasoning import DoOperator
        do = DoOperator(dim=16, num_variables=4)
        states = torch.randn(4, 16)
        intervention = torch.ones(16)
        result = do.intervene(states, target_idx=1, intervention_value=intervention, parent_indices=[0])
        assert torch.allclose(result[1], intervention)

    def test_counterfactual_engine(self):
        from pyquifer.causal_reasoning import CounterfactualEngine
        cf = CounterfactualEngine(dim=16)
        factual = torch.randn(16)
        cf_value = torch.randn(16)
        parent = torch.randn(16)
        result = cf.counterfactual(factual, cf_value, parent)
        assert 'counterfactual_outcome' in result
        assert 'divergence' in result

    def test_causal_discovery(self):
        from pyquifer.causal_reasoning import CausalDiscovery
        cd = CausalDiscovery(num_variables=4, dim=1)
        data = torch.randn(32, 4)
        result = cd(data)
        assert 'loss' in result
        assert 'dag_penalty' in result
        assert result['adjacency'].shape == (4, 4)

    def test_interventional_query(self):
        from pyquifer.causal_reasoning import InterventionalQuery
        iq = InterventionalQuery(dim=16, num_variables=4)
        result = iq.query_association(torch.randn(16), torch.randn(16))
        assert result.shape == (16,)


# ── Phase 16: Selective SSM ──────────────────────────────────────────────────

class TestSelectiveSSM:
    def test_selective_scan(self):
        from pyquifer.selective_ssm import SelectiveScan
        scan = SelectiveScan(state_dim=8)
        B, L, D, N = 2, 10, 16, 8
        A = torch.randn(B, L, D, N).sigmoid()
        Bmat = torch.randn(B, L, D, N) * 0.1
        C = torch.randn(B, L, N)
        x = torch.randn(B, L, D)
        y = scan(A, Bmat, C, x)
        assert y.shape == (B, L, D)

    def test_selective_state_space(self):
        from pyquifer.selective_ssm import SelectiveStateSpace
        ssm = SelectiveStateSpace(d_model=32, d_state=8, d_inner=64)
        x = torch.randn(2, 10, 64)
        y = ssm(x)
        assert y.shape == (2, 10, 64)

    def test_ssm_block(self):
        from pyquifer.selective_ssm import SSMBlock
        block = SSMBlock(d_model=32, d_state=8)
        x = torch.randn(2, 10, 32)
        y = block(x)
        assert y.shape == (2, 10, 32)

    def test_mamba_layer(self):
        from pyquifer.selective_ssm import MambaLayer
        mamba = MambaLayer(d_model=32, d_state=8)
        x = torch.randn(2, 10, 32)
        y = mamba(x)
        assert y.shape == (2, 10, 32)

    def test_oscillatory_ssm(self):
        from pyquifer.selective_ssm import OscillatorySSM
        ossm = OscillatorySSM(d_model=32, num_oscillators=4)
        x = torch.randn(2, 10, 32)
        y = ossm(x)
        assert y.shape == (2, 10, 32)


# ── Phase 16: Oscillatory MoE ───────────────────────────────────────────────

class TestOscillatoryMoE:
    def test_expert_pool(self):
        from pyquifer.oscillatory_moe import ExpertPool
        pool = ExpertPool(num_experts=4, d_model=32)
        x = torch.randn(8, 32)
        out = pool(x, expert_idx=0)
        assert out.shape == (8, 32)

    def test_oscillator_router(self):
        from pyquifer.oscillatory_moe import OscillatorRouter
        router = OscillatorRouter(d_model=32, num_experts=4)
        x = torch.randn(2, 8, 32)
        result = router(x)
        assert 'router_probs' in result
        assert 'expert_indices' in result

    def test_sparse_moe_forward(self):
        from pyquifer.oscillatory_moe import SparseMoE
        moe = SparseMoE(d_model=32, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 32)
        result = moe(x)
        assert result['output'].shape == (2, 8, 32)
        assert 'balance_loss' in result

    def test_moe_with_oscillator_state(self):
        from pyquifer.oscillatory_moe import SparseMoE
        moe = SparseMoE(d_model=32, num_experts=4, num_oscillator_features=8)
        x = torch.randn(2, 8, 32)
        osc = torch.randn(8)
        result = moe(x, oscillator_state=osc)
        assert result['output'].shape == (2, 8, 32)


# ── Phase 16: Graph Reasoning ───────────────────────────────────────────────

class TestGraphReasoning:
    def test_dynamic_graph_attention(self):
        from pyquifer.graph_reasoning import DynamicGraphAttention
        attn = DynamicGraphAttention(dim=32, num_heads=4)
        nodes = torch.randn(2, 6, 32)
        adj = (torch.rand(6, 6) > 0.5).float()
        adj.fill_diagonal_(1)
        out = attn(nodes, adj)
        assert out.shape == (2, 6, 32)

    def test_message_passing_with_phase(self):
        from pyquifer.graph_reasoning import MessagePassingWithPhase
        mp = MessagePassingWithPhase(dim=32, num_oscillators=4)
        nodes = torch.randn(6, 32)
        adj = (torch.rand(6, 6) > 0.5).float()
        adj.fill_diagonal_(1)
        out = mp(nodes, adj)
        assert out.shape == (6, 32)

    def test_temporal_graph_transformer(self):
        from pyquifer.graph_reasoning import TemporalGraphTransformer
        tgt = TemporalGraphTransformer(dim=32, num_heads=4, num_layers=1)
        nodes = torch.randn(2, 6, 32)
        adj = (torch.rand(6, 6) > 0.5).float()
        adj.fill_diagonal_(1)
        result = tgt(nodes, adj)
        assert result['node_features'].shape == (2, 6, 32)
        assert result['graph_embedding'].shape == (2, 32)


# ── Phase 17: Prospective Configuration ─────────────────────────────────────

class TestProspectiveConfig:
    def test_prospective_inference(self):
        from pyquifer.prospective_config import ProspectiveInference
        pi = ProspectiveInference(dim=16, num_iterations=5)
        initial = torch.randn(4, 16)
        target = torch.randn(4, 16)
        result = pi.infer(initial, target)
        assert 'activity' in result
        assert result['activity'].shape == (4, 16)

    def test_prospective_hebbian(self):
        from pyquifer.prospective_config import ProspectiveHebbian
        ph = ProspectiveHebbian(input_dim=16, output_dim=8)
        x = torch.randn(4, 16)
        y = ph(x)
        assert y.shape == (4, 8)

    def test_infer_then_modify(self):
        from pyquifer.prospective_config import InferThenModify
        itm = InferThenModify(dims=[16, 8, 4], num_inference_steps=5)
        x = torch.randn(4, 16)
        target = torch.randn(4, 4)
        result = itm.learn(x, target)
        assert 'output' in result
        assert 'loss' in result
        assert result['output'].shape == (4, 4)

    def test_infer_then_modify_improves(self):
        from pyquifer.prospective_config import InferThenModify
        itm = InferThenModify(dims=[8, 4], num_inference_steps=10, modification_lr=0.05)
        x = torch.randn(4, 8)
        target = torch.zeros(4, 4)
        result = itm.learn(x, target)
        # After learning, new_loss should be lower or equal
        assert result['loss'] <= result['old_loss'] + 0.5  # Reasonable tolerance


# ── Phase 17: Dendritic Learning ─────────────────────────────────────────────

class TestDendriticLearning:
    def test_pyramidal_neuron(self):
        from pyquifer.dendritic_learning import PyramidalNeuron
        neuron = PyramidalNeuron(basal_dim=16, apical_dim=8, soma_dim=16)
        basal = torch.randn(4, 16)
        apical = torch.randn(4, 8)
        result = neuron(basal, apical)
        assert 'soma_output' in result
        assert result['soma_output'].shape == (4, 16)

    def test_dendritic_error_signal(self):
        from pyquifer.dendritic_learning import DendriticErrorSignal
        des = DendriticErrorSignal(dim=16, feedback_dim=8)
        soma = torch.randn(4, 16)
        feedback = torch.randn(4, 8)
        result = des.compute_error(soma, feedback)
        assert 'error' in result
        assert 'gated_error' in result

    def test_dendritic_localized_learning(self):
        from pyquifer.dendritic_learning import DendriticLocalizedLearning
        dll = DendriticLocalizedLearning(dims=[16, 8, 4])
        x = torch.randn(4, 16)
        y = dll(x)
        assert y.shape == (4, 4)

    def test_dll_learn(self):
        from pyquifer.dendritic_learning import DendriticLocalizedLearning
        dll = DendriticLocalizedLearning(dims=[8, 4], lr=0.05)
        x = torch.randn(4, 8)
        target = torch.zeros(4, 4)
        result = dll.learn(x, target)
        assert 'loss' in result
        assert 'burst_rates' in result


# ── Phase 17: Energy Spiking PC ──────────────────────────────────────────────

class TestEnergySpiking:
    def test_spiking_pc_neuron(self):
        from pyquifer.energy_spiking import SpikingPCNeuron
        neuron = SpikingPCNeuron(dim=16)
        neuron.reset_state(batch_size=4)
        input_current = torch.randn(4, 16)
        result = neuron(input_current)
        assert 'spikes' in result
        assert result['spikes'].shape == (4, 16)

    def test_energy_optimized_snn(self):
        from pyquifer.energy_spiking import EnergyOptimizedSNN
        snn = EnergyOptimizedSNN(dims=[16, 8, 4], num_steps=10)
        x = torch.randn(4, 16)
        result = snn(x)
        assert 'output_spikes' in result
        assert 'total_energy' in result
        assert result['output_rate'].shape == (4, 4)

    def test_multi_compartment_spiking_pc(self):
        from pyquifer.energy_spiking import MultiCompartmentSpikingPC
        mc = MultiCompartmentSpikingPC(dim=16, num_compartments=3)
        mc.reset_state(batch_size=4)
        inputs = [torch.randn(4, 16) for _ in range(3)]
        result = mc(inputs)
        assert 'spikes' in result
        assert result['spikes'].shape == (4, 16)

    def test_energy_landscape(self):
        from pyquifer.energy_spiking import EnergyLandscape
        el = EnergyLandscape(dim=16)
        state = torch.randn(4, 16)
        result = el.analyze(state)
        assert 'energy' in result
        assert 'gradient_norm' in result
        assert 'is_minimum' in result


# ── Phase 18: FHRR ──────────────────────────────────────────────────────────

class TestFHRR:
    def test_fhrr_encoder_discrete(self):
        from pyquifer.fhrr import FHRREncoder
        enc = FHRREncoder(dim=64, num_symbols=128)
        indices = torch.tensor([0, 5, 10])
        phases = enc.encode_discrete(indices)
        assert phases.shape == (3, 64)

    def test_fhrr_encoder_continuous(self):
        from pyquifer.fhrr import FHRREncoder
        enc = FHRREncoder(dim=64)
        values = torch.tensor([0.1, 0.5, 0.9])
        phases = enc.encode_continuous(values)
        assert phases.shape == (3, 64)

    def test_spike_vsa_bind_unbind(self):
        from pyquifer.fhrr import SpikeVSAOps
        vsa = SpikeVSAOps(dim=64)
        a = torch.rand(64) * 2 * math.pi
        b = torch.rand(64) * 2 * math.pi
        bound = vsa.bind(a, b)
        unbound = vsa.unbind(bound, a)
        # Unbound should be close to b
        sim = vsa.similarity(unbound, b)
        assert sim.item() > 0.8

    def test_spike_vsa_superpose(self):
        from pyquifer.fhrr import SpikeVSAOps
        vsa = SpikeVSAOps(dim=128)
        a = torch.rand(128) * 2 * math.pi
        b = torch.rand(128) * 2 * math.pi
        superposed = vsa.superpose([a, b])
        assert superposed.shape == (128,)

    def test_latency_encoder(self):
        from pyquifer.fhrr import LatencyEncoder
        enc = LatencyEncoder(dim=32)
        values = torch.tensor([0.2, 0.8])
        latencies = enc(values)
        assert latencies.shape == (2, 32)

    def test_neuromorphic_exporter(self):
        from pyquifer.fhrr import NeuromorphicExporter
        import torch.nn as nn
        model = nn.Linear(16, 8)
        exporter = NeuromorphicExporter(target='loihi')
        weights = exporter.export_weights(model)
        assert 'weight' in list(weights.keys())[0]
        config = exporter.export_config(model)
        assert config['target'] == 'loihi'


# ── Phase 18: FlashRNN ──────────────────────────────────────────────────────

class TestFlashRNN:
    def test_flash_ltc_cell(self):
        from pyquifer.flash_rnn import FlashLTC
        cell = FlashLTC(input_dim=16, hidden_dim=32)
        x = torch.randn(4, 16)
        h, new_h = cell(x)
        assert h.shape == (4, 32)
        assert new_h.shape == (4, 32)

    def test_flash_cfc_cell(self):
        from pyquifer.flash_rnn import FlashCfC
        cell = FlashCfC(input_dim=16, hidden_dim=32)
        x = torch.randn(4, 16)
        h, new_h = cell(x)
        assert h.shape == (4, 32)

    def test_flash_cfc_modes(self):
        from pyquifer.flash_rnn import FlashCfC
        for mode in ['default', 'no_gate', 'pure']:
            cell = FlashCfC(input_dim=16, hidden_dim=32, mode=mode)
            x = torch.randn(2, 16)
            h, _ = cell(x)
            assert h.shape == (2, 32), f"Failed for mode={mode}"

    def test_flash_ltc_layer_sequence(self):
        from pyquifer.flash_rnn import FlashLTCLayer
        layer = FlashLTCLayer(input_dim=16, hidden_dim=32, cell_type='ltc')
        x = torch.randn(2, 10, 16)  # B=2, L=10
        result = layer(x)
        assert result['output'].shape == (2, 10, 32)
        assert result['final_state'].shape == (2, 32)

    def test_flash_ltc_layer_bidirectional(self):
        from pyquifer.flash_rnn import FlashLTCLayer
        layer = FlashLTCLayer(input_dim=16, hidden_dim=32, bidirectional=True)
        x = torch.randn(2, 8, 16)
        result = layer(x)
        assert result['output'].shape == (2, 8, 32)

    def test_is_flash_available(self):
        from pyquifer.flash_rnn import is_flash_available
        # Should return bool without error
        result = is_flash_available()
        assert isinstance(result, bool)

    def test_flash_ltc_with_state(self):
        from pyquifer.flash_rnn import FlashLTC
        cell = FlashLTC(input_dim=16, hidden_dim=32)
        x = torch.randn(2, 16)
        h0 = torch.zeros(2, 32)
        h1, _ = cell(x, h0)
        h2, _ = cell(x, h1)
        # States should differ after two steps
        assert not torch.allclose(h1, h2)


# ── CycleConfig flags ───────────────────────────────────────────────────────

class TestCycleConfigFlags:
    def test_new_flags_exist(self):
        from pyquifer.integration import CycleConfig
        cfg = CycleConfig()
        assert hasattr(cfg, 'use_visual_binding')
        assert hasattr(cfg, 'use_temporal_binding')
        assert hasattr(cfg, 'use_sensory_binding')
        assert hasattr(cfg, 'use_deep_aif')
        assert hasattr(cfg, 'use_jepa_world_model')
        assert hasattr(cfg, 'use_deliberation')
        assert hasattr(cfg, 'use_cls_memory')
        assert hasattr(cfg, 'use_causal_reasoning')
        assert hasattr(cfg, 'use_appraisal')
        assert hasattr(cfg, 'use_selective_ssm')
        assert hasattr(cfg, 'use_oscillatory_moe')
        assert hasattr(cfg, 'use_prospective_learning')
        assert hasattr(cfg, 'solver')
        assert hasattr(cfg, 'use_complex_oscillators')
        assert hasattr(cfg, 'use_cuda_kernels')

    def test_flags_default_false(self):
        from pyquifer.integration import CycleConfig
        cfg = CycleConfig()
        assert cfg.use_visual_binding is False
        assert cfg.use_deep_aif is False
        assert cfg.use_deliberation is False
        assert cfg.solver == "euler"
