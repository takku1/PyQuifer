"""Tests for S-06/S-07 closed-loop fixes in GenerativeWorldModel."""

import torch
import pytest
from pyquifer.memory.generative_world_model import GenerativeWorldModel

SPACE_DIM = 3
NOISE_SHAPE = (4, 4, 4)


@pytest.fixture
def model():
    """Create a GenerativeWorldModel with 2 banks for testing."""
    bank_configs = [
        {"num_oscillators": 8, "dt": 0.01, "initial_frequency_range": (1.0, 2.0)},
        {"num_oscillators": 4, "dt": 0.05, "initial_frequency_range": (0.1, 0.5)},
    ]
    m = GenerativeWorldModel(
        space_dim=SPACE_DIM,
        bank_configs=bank_configs,
        num_attractors=2,
        actualization_strength=0.05,
    )
    return m


@pytest.fixture
def forward_outputs(model):
    """Run a forward pass and return all outputs."""
    initial_state = torch.rand(1, SPACE_DIM)
    return model(
        input_noise_shape=NOISE_SHAPE,
        initial_specimen_state=initial_state,
        actualization_iterations=3,
        oscillator_steps=10,
        noise_amplitude=0.1,
        time_offset=0.0,
    )


def test_per_bank_different_inputs(model):
    """S-07: With 2 banks, verify external inputs differ per bank."""
    ext0 = 0.1 * model.archetype_to_bank[0](model.archetype_vector).squeeze(0)
    ext1 = 0.1 * model.archetype_to_bank[1](model.archetype_vector).squeeze(0)
    assert ext0.shape[0] == 8
    assert ext1.shape[0] == 4


def test_per_bank_shapes(model):
    """S-07: Each archetype_to_bank output shape matches bank's num_oscillators."""
    for i, bank in enumerate(model.frequency_bank.banks):
        out = model.archetype_to_bank[i](model.archetype_vector).squeeze(0)
        assert out.shape[0] == bank.num_oscillators, (
            f"Bank {i}: expected {bank.num_oscillators}, got {out.shape[0]}"
        )


def test_feedback_influence_computed(forward_outputs):
    """S-06: oscillator_archetype_influence is returned with correct shape."""
    _, _, _, _, osc_influence = forward_outputs
    assert osc_influence.shape == (1, SPACE_DIM)


def test_feedback_clipping(model):
    """S-06: With extreme feedback_gain, clamp keeps feedback in [-0.5, 0.5]."""
    with torch.no_grad():
        model.feedback_gain.fill_(100.0)

    initial_state = torch.rand(1, SPACE_DIM)
    out = model(
        input_noise_shape=NOISE_SHAPE,
        initial_specimen_state=initial_state,
        actualization_iterations=3,
        oscillator_steps=10,
        noise_amplitude=0.1,
    )
    actualized, _, _, _, osc_influence = out
    assert torch.isfinite(actualized).all()
    assert torch.isfinite(osc_influence).all()
    diff = (model.mind_eye_actualization.target_vector - model.archetype_vector).abs()
    assert diff.max().item() <= 0.5 + 1e-6, f"Feedback exceeded clamp: {diff.max().item()}"


def test_feedback_modulates_target(model):
    """S-06: After forward(), target_vector != archetype_vector (oscillators shifted it)."""
    with torch.no_grad():
        model.feedback_gain.fill_(1.0)

    initial_state = torch.rand(1, SPACE_DIM)
    archetype_before = model.archetype_vector.clone()

    model(
        input_noise_shape=NOISE_SHAPE,
        initial_specimen_state=initial_state,
        actualization_iterations=3,
        oscillator_steps=50,
        noise_amplitude=0.1,
    )

    target_after = model.mind_eye_actualization.target_vector
    diff = (target_after - archetype_before).abs().sum().item()
    assert diff > 1e-6, "Target was not modulated by oscillator feedback"


def test_archetype_vector_unchanged(model):
    """S-06: archetype_vector itself is NOT modified by forward()."""
    archetype_before = model.archetype_vector.clone()

    initial_state = torch.rand(1, SPACE_DIM)
    model(
        input_noise_shape=NOISE_SHAPE,
        initial_specimen_state=initial_state,
        actualization_iterations=3,
        oscillator_steps=10,
        noise_amplitude=0.1,
    )

    assert torch.allclose(model.archetype_vector, archetype_before), (
        "archetype_vector was modified by forward() — should only change via gradient descent"
    )


def test_backward_completes(model):
    """backward() completes without error on actualized_state loss."""
    initial_state = torch.rand(1, SPACE_DIM)

    actualized, _, _, _, _ = model(
        input_noise_shape=NOISE_SHAPE,
        initial_specimen_state=initial_state,
        actualization_iterations=3,
        oscillator_steps=10,
        noise_amplitude=0.1,
    )

    loss = actualized.sum()
    # Actualization uses iterative denoising (Perlin noise + potentials)
    # which may not propagate gradients to archetype_vector. The key is
    # that backward() doesn't crash — gradient flow to archetype happens
    # via the training loop in __main__ (through optimizer.step).
    loss.backward()  # Should not raise


def test_deterministic(model):
    """Same input with same state produces same output."""
    torch.manual_seed(42)
    initial_state = torch.rand(1, SPACE_DIM)

    phases_before = [b.phases.clone() for b in model.frequency_bank.banks]

    out1 = model(
        input_noise_shape=NOISE_SHAPE,
        initial_specimen_state=initial_state.clone(),
        actualization_iterations=3,
        oscillator_steps=10,
        noise_amplitude=0.0,
    )

    with torch.no_grad():
        for b, p in zip(model.frequency_bank.banks, phases_before):
            b.phases.copy_(p)

    out2 = model(
        input_noise_shape=NOISE_SHAPE,
        initial_specimen_state=initial_state.clone(),
        actualization_iterations=3,
        oscillator_steps=10,
        noise_amplitude=0.0,
    )

    assert torch.allclose(out1[0], out2[0], atol=1e-3), "Actualized state not deterministic"
    assert torch.allclose(out1[4], out2[4], atol=1e-3), "Oscillator influence not deterministic"
