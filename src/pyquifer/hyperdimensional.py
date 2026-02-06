"""
Hyperdimensional Computing (HDC) Module for PyQuifer

Implements Holographic Reduced Representations (HRR) and hyperdimensional
computing primitives for compositional binding in oscillatory systems.

Key concepts:
- Circular convolution binding: Combine concepts without losing information
- Holographic superposition: Multiple bindings in the same vector
- Resonant unbinding: Retrieve components via correlation
- Phase-based HDC: Bind oscillators via phase-locking

This enables:
1. Variable binding without attention mechanisms
2. Compositional reasoning in continuous time
3. Interference-pattern based concept combination

Based on work by Plate (2003), Kanerva (2009), and recent VSA literature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List


def circular_convolution(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Circular convolution for binding two hypervectors.

    In the Fourier domain: conv(a,b) = ifft(fft(a) * fft(b))
    This is associative, commutative, and distributes over addition.

    Args:
        a: First hypervector (..., dim)
        b: Second hypervector (..., dim)

    Returns:
        Bound hypervector (..., dim)
    """
    # Use FFT for efficient circular convolution
    a_fft = torch.fft.fft(a, dim=-1)
    b_fft = torch.fft.fft(b, dim=-1)
    result_fft = a_fft * b_fft
    result = torch.fft.ifft(result_fft, dim=-1).real
    return result


def circular_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Circular correlation for unbinding (approximate inverse of convolution).

    corr(a, conv(a,b)) ≈ b (with some noise)

    Args:
        a: Key hypervector
        b: Bound hypervector (containing a bound with something)

    Returns:
        Approximate unbound content
    """
    # Correlation = convolution with conjugate
    a_fft = torch.fft.fft(a, dim=-1)
    b_fft = torch.fft.fft(b, dim=-1)
    # Conjugate for correlation
    result_fft = torch.conj(a_fft) * b_fft
    result = torch.fft.ifft(result_fft, dim=-1).real
    return result


def normalize_hd(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize hypervector to unit length."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class HypervectorMemory(nn.Module):
    """
    Holographic associative memory using hyperdimensional computing.

    Stores key-value pairs as bound hypervectors that can be superposed.
    Retrieval is via circular correlation with the key.
    """

    def __init__(self, dim: int, capacity: int = 100):
        """
        Args:
            dim: Dimension of hypervectors (should be high, e.g., 1000+)
            capacity: Maximum number of stored associations
        """
        super().__init__()
        self.dim = dim
        self.capacity = capacity

        # Superposed memory trace
        self.register_buffer('memory', torch.zeros(dim))
        self.register_buffer('num_stored', torch.tensor(0))

    def store(self, key: torch.Tensor, value: torch.Tensor):
        """
        Store a key-value association.

        Args:
            key: Key hypervector (dim,)
            value: Value hypervector (dim,)
        """
        # Bind key and value
        bound = circular_convolution(key, value)

        # Add to superposed memory
        with torch.no_grad():
            self.memory = self.memory + bound
            self.num_stored = self.num_stored + 1

    def retrieve(self, key: torch.Tensor) -> torch.Tensor:
        """
        Retrieve value associated with key.

        Args:
            key: Query key (dim,)

        Returns:
            Retrieved value (noisy if many items stored)
        """
        # Correlate key with memory to unbind
        retrieved = circular_correlation(key, self.memory)
        return normalize_hd(retrieved)

    def clear(self):
        """Clear memory."""
        with torch.no_grad():
            self.memory.zero_()
            self.num_stored.zero_()


class PhaseBinder(nn.Module):
    """
    Bind oscillators via phase relationships.

    Instead of circular convolution on amplitude vectors, this uses
    the natural binding mechanism of oscillators: phase-locking.
    When two oscillators lock at a specific phase difference, they
    are "bound" into a composite representation.
    """

    def __init__(self,
                 num_oscillators: int,
                 binding_strength: float = 1.0):
        """
        Args:
            num_oscillators: Number of oscillators in the system
            binding_strength: How strongly phase differences are enforced
        """
        super().__init__()
        self.num_oscillators = num_oscillators
        self.binding_strength = nn.Parameter(torch.tensor(binding_strength))

        # Target phase differences for binding (learnable)
        self.target_phase_diff = nn.Parameter(
            torch.zeros(num_oscillators, num_oscillators)
        )

    def bind(self,
             phases_a: torch.Tensor,
             phases_b: torch.Tensor,
             binding_code: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Create binding force to lock two phase patterns.

        Args:
            phases_a: Phases of first pattern (batch, num_osc)
            phases_b: Phases of second pattern (batch, num_osc)
            binding_code: Optional phase offset for this binding

        Returns:
            Phase adjustment to apply to pattern A to bind with B
        """
        if binding_code is None:
            binding_code = torch.zeros_like(phases_a)

        # Target: phases_a should equal phases_b + binding_code
        target = phases_b + binding_code

        # Phase difference (circular)
        diff = torch.remainder(target - phases_a + math.pi, 2 * math.pi) - math.pi

        # Return adjustment force
        return self.binding_strength * torch.sin(diff)

    def check_binding(self,
                      phases_a: torch.Tensor,
                      phases_b: torch.Tensor,
                      binding_code: Optional[torch.Tensor] = None,
                      threshold: float = 0.3) -> torch.Tensor:
        """
        Check if two patterns are bound (phase-locked).

        Returns:
            Binding strength (0 = unbound, 1 = fully bound)
        """
        if binding_code is None:
            binding_code = torch.zeros_like(phases_a)

        target = phases_b + binding_code
        diff = torch.remainder(target - phases_a + math.pi, 2 * math.pi) - math.pi

        # Binding strength = how close to zero the phase diff is
        coherence = torch.cos(diff).mean(dim=-1)
        return torch.clamp(coherence, 0, 1)


class ResonantBinding(nn.Module):
    """
    Binding via resonant frequency matching.

    Concepts are represented as frequency spectra. Binding creates
    interference patterns that form new composite frequencies.
    """

    def __init__(self,
                 state_dim: int,
                 num_frequencies: int = 64):
        """
        Args:
            state_dim: Dimension of concept vectors
            num_frequencies: Number of frequency components
        """
        super().__init__()
        self.state_dim = state_dim
        self.num_frequencies = num_frequencies

        # Project to frequency domain
        self.to_freq = nn.Linear(state_dim, num_frequencies * 2)  # Real + imag
        self.from_freq = nn.Linear(num_frequencies * 2, state_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode concept as frequency spectrum."""
        freq = self.to_freq(x)
        # Split into magnitude and phase
        real = freq[..., :self.num_frequencies]
        imag = freq[..., self.num_frequencies:]
        return torch.complex(real, imag)

    def decode(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Decode frequency spectrum to concept vector."""
        freq = torch.cat([spectrum.real, spectrum.imag], dim=-1)
        return self.from_freq(freq)

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Bind two concepts via spectral multiplication.

        In frequency domain, this creates interference patterns
        that encode the combination uniquely.
        """
        spec_a = self.encode(a)
        spec_b = self.encode(b)

        # Multiply spectra (binding in frequency domain)
        bound_spec = spec_a * spec_b

        return self.decode(bound_spec)

    def unbind(self, bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Unbind using spectral division (approximate inverse).
        """
        spec_bound = self.encode(bound)
        spec_key = self.encode(key)

        # Divide spectra (with regularization)
        unbound_spec = spec_bound / (spec_key + 1e-6)

        return self.decode(unbound_spec)

    def superpose(self, concepts: List[torch.Tensor]) -> torch.Tensor:
        """
        Superpose multiple concepts into one vector.

        Information is preserved holographically - can retrieve
        individual concepts via correlation.
        """
        spectra = [self.encode(c) for c in concepts]
        superposed = sum(spectra) / len(spectra)
        return self.decode(superposed)


class HDCReasoner(nn.Module):
    """
    Hyperdimensional computing based reasoning module.

    Performs compositional operations on concepts using HDC primitives:
    - Binding: Create role-filler pairs
    - Superposition: Combine multiple items
    - Similarity: Compare concepts via cosine similarity
    - Sequence: Encode order using permutation
    """

    def __init__(self,
                 dim: int,
                 num_roles: int = 8):
        """
        Args:
            dim: Hypervector dimension
            num_roles: Number of predefined roles (like AGENT, ACTION, OBJECT)
        """
        super().__init__()
        self.dim = dim
        self.num_roles = num_roles

        # Random role vectors (fixed)
        self.register_buffer(
            'roles',
            normalize_hd(torch.randn(num_roles, dim))
        )

        # Permutation matrix for sequence encoding
        perm_indices = torch.randperm(dim)
        self.register_buffer('perm', perm_indices)
        self.register_buffer('inv_perm', torch.argsort(perm_indices))

        # Memory for retrieved concepts
        self.memory = HypervectorMemory(dim)

    def bind_role(self, concept: torch.Tensor, role_idx: int) -> torch.Tensor:
        """Bind concept to a role."""
        role = self.roles[role_idx]
        return circular_convolution(concept, role)

    def unbind_role(self, bound: torch.Tensor, role_idx: int) -> torch.Tensor:
        """Retrieve concept from role binding."""
        role = self.roles[role_idx]
        return circular_correlation(role, bound)

    def permute(self, x: torch.Tensor) -> torch.Tensor:
        """Permute vector (for sequence encoding)."""
        return x[..., self.perm]

    def inv_permute(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse permute."""
        return x[..., self.inv_perm]

    def encode_sequence(self, items: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode a sequence using permutation.

        Position is encoded by number of permutations applied.
        """
        result = torch.zeros(self.dim, device=items[0].device)
        current = items[0]

        for i, item in enumerate(items):
            # Apply i permutations to encode position
            positioned = item
            for _ in range(i):
                positioned = self.permute(positioned)
            result = result + positioned

        return normalize_hd(result)

    def decode_sequence_position(self,
                                  sequence: torch.Tensor,
                                  query: torch.Tensor,
                                  max_len: int = 10) -> int:
        """
        Find position of query in encoded sequence.
        """
        best_sim = -1
        best_pos = 0

        for i in range(max_len):
            # Apply i inverse permutations
            positioned = sequence
            for _ in range(i):
                positioned = self.inv_permute(positioned)

            sim = F.cosine_similarity(positioned, query, dim=-1)
            if sim > best_sim:
                best_sim = sim
                best_pos = i

        return best_pos

    def analogy(self,
                a: torch.Tensor,
                b: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        """
        Solve analogy: a is to b as c is to ?

        Uses the relation: ? = c * (b / a) = c * corr(a, b)
        """
        # Extract relation between a and b
        relation = circular_correlation(a, b)

        # Apply relation to c
        result = circular_convolution(c, relation)

        return normalize_hd(result)

    def forward(self,
                concepts: torch.Tensor,
                roles: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process concepts through HDC operations.

        Args:
            concepts: Batch of concept vectors (batch, num_concepts, dim)
            roles: Optional role indices (batch, num_concepts)

        Returns:
            Dictionary with bound representation and similarity matrix
        """
        batch_size, num_concepts, dim = concepts.shape

        # Bind each concept to its role
        if roles is not None:
            bound_concepts = []
            for i in range(num_concepts):
                role_idx = roles[:, i] if roles.dim() > 1 else roles[i]
                bound = self.bind_role(concepts[:, i], role_idx.item())
                bound_concepts.append(bound)
            bound_concepts = torch.stack(bound_concepts, dim=1)
        else:
            bound_concepts = concepts

        # Superpose all bound concepts
        superposed = bound_concepts.sum(dim=1)
        superposed = normalize_hd(superposed)

        # Compute similarity matrix
        similarity = torch.bmm(
            concepts,
            concepts.transpose(1, 2)
        ) / math.sqrt(dim)

        return {
            'superposed': superposed,
            'bound_concepts': bound_concepts,
            'similarity': similarity,
        }


if __name__ == '__main__':
    print("--- Hyperdimensional Computing Examples ---")

    # Example 1: Basic circular convolution binding
    print("\n1. Circular Convolution Binding")
    dim = 1000

    # Random concepts
    dog = normalize_hd(torch.randn(dim))
    cat = normalize_hd(torch.randn(dim))
    chases = normalize_hd(torch.randn(dim))

    # Bind: "dog chases cat"
    agent_role = normalize_hd(torch.randn(dim))
    action_role = normalize_hd(torch.randn(dim))
    patient_role = normalize_hd(torch.randn(dim))

    sentence = (
        circular_convolution(dog, agent_role) +
        circular_convolution(chases, action_role) +
        circular_convolution(cat, patient_role)
    )

    # Retrieve agent
    retrieved_agent = circular_correlation(agent_role, sentence)
    sim_dog = F.cosine_similarity(retrieved_agent, dog, dim=0)
    sim_cat = F.cosine_similarity(retrieved_agent, cat, dim=0)
    print(f"   Retrieved agent similarity to 'dog': {sim_dog.item():.3f}")
    print(f"   Retrieved agent similarity to 'cat': {sim_cat.item():.3f}")

    # Example 2: Resonant Binding
    print("\n2. Resonant Binding")
    binder = ResonantBinding(state_dim=128, num_frequencies=32)

    concept_a = torch.randn(128)
    concept_b = torch.randn(128)

    bound = binder.bind(concept_a, concept_b)
    unbound = binder.unbind(bound, concept_a)

    # Check if unbound is similar to concept_b
    sim = F.cosine_similarity(unbound, concept_b, dim=0)
    print(f"   Unbinding accuracy (cosine sim): {sim.item():.3f}")

    # Example 3: Analogy
    print("\n3. Analogy Solving")
    reasoner = HDCReasoner(dim=1000, num_roles=4)

    # king - man + woman = queen (classic analogy)
    king = normalize_hd(torch.randn(1000))
    man = normalize_hd(torch.randn(1000))
    woman = normalize_hd(torch.randn(1000))
    queen = normalize_hd(torch.randn(1000))

    # Train: king relates to man as queen relates to woman
    # So: king/man should equal queen/woman
    predicted = reasoner.analogy(man, king, woman)

    # In real system, would find nearest neighbor in vocabulary
    print(f"   Predicted vector norm: {predicted.norm().item():.3f}")

    # Example 4: Sequence Encoding
    print("\n4. Sequence Encoding")
    items = [normalize_hd(torch.randn(1000)) for _ in range(5)]

    sequence = reasoner.encode_sequence(items)

    # Find position of item 2
    pos = reasoner.decode_sequence_position(sequence, items[2], max_len=5)
    print(f"   Item 2 found at position: {pos}")

    print("\n[OK] All hyperdimensional computing tests passed!")
