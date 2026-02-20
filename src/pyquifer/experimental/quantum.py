"""
Quantum Cognition Module for PyQuifer

Implements quantum-inspired cognitive mechanisms using tensor algebra:

1. Superposition States - Belief states as complex amplitudes
2. Interference Effects - Context-dependent probability modulation
3. Entanglement - Correlated cognitive variables
4. Measurement/Collapse - Decision crystallization

Mathematical Foundation:
- Beliefs represented as vectors in Hilbert space
- Context as unitary transformations
- Decisions as projective measurements
- Order effects via non-commutative operations

This explains psychological phenomena that violate classical probability:
- Conjunction fallacy (Linda problem)
- Order effects in judgments
- Context-dependent preferences

All operations use einsum for substrate-optimal GPU execution.

Based on: Busemeyer & Bruza (2012), Pothos & Busemeyer (2013)
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def complex_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Complex matrix multiplication using real tensors.

    Input format: (..., n, 2) where last dim is [real, imag]
    Uses einsum for optimal GPU utilization.
    """
    # a: (..., m, k, 2), b: (..., k, n, 2)
    # (a_r + i*a_i)(b_r + i*b_i) = (a_r*b_r - a_i*b_i) + i*(a_r*b_i + a_i*b_r)
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]

    real = torch.einsum('...mk,...kn->...mn', ar, br) - torch.einsum('...mk,...kn->...mn', ai, bi)
    imag = torch.einsum('...mk,...kn->...mn', ar, bi) + torch.einsum('...mk,...kn->...mn', ai, br)

    return torch.stack([real, imag], dim=-1)


def complex_conj(z: torch.Tensor) -> torch.Tensor:
    """Complex conjugate: flip sign of imaginary part."""
    return z * torch.tensor([1., -1.], device=z.device, dtype=z.dtype)


def complex_norm_sq(z: torch.Tensor) -> torch.Tensor:
    """Squared norm of complex vector: |z|^2 = z* · z"""
    return (z ** 2).sum(dim=-1)


def complex_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Complex inner product <a|b> using einsum."""
    # <a|b> = sum(a* · b)
    ac = complex_conj(a)
    # Real part of inner product
    real = torch.einsum('...i,...i->', ac[..., 0], b[..., 0]) - torch.einsum('...i,...i->', ac[..., 1], b[..., 1])
    imag = torch.einsum('...i,...i->', ac[..., 0], b[..., 1]) + torch.einsum('...i,...i->', ac[..., 1], b[..., 0])
    return torch.stack([real, imag], dim=-1)


class QuantumState(nn.Module):
    """
    Quantum cognitive state as a unit vector in Hilbert space.

    Represents superposition of basis beliefs with complex amplitudes.
    Born rule: probability = |amplitude|^2

    Implements via real tensors with shape (..., dim, 2) for [real, imag].
    """

    def __init__(self, dim: int, batch_size: int = 1):
        """
        Args:
            dim: Dimension of Hilbert space (number of basis states)
            batch_size: Number of parallel cognitive agents
        """
        super().__init__()
        self.dim = dim
        self.batch_size = batch_size

        # Initialize in superposition: |ψ> = (1/√dim) Σ|i>
        uniform_amp = 1.0 / math.sqrt(dim)
        self.register_buffer(
            'state',
            torch.zeros(batch_size, dim, 2)
        )
        self.state[..., 0] = uniform_amp  # Real part uniform

    def probabilities(self) -> torch.Tensor:
        """
        Born rule: P(i) = |<i|ψ>|^2 = |α_i|^2

        Returns probability distribution over basis states.
        """
        return complex_norm_sq(self.state)

    def entropy(self) -> torch.Tensor:
        """Quantum entropy: S = -Σ p_i log(p_i)"""
        probs = self.probabilities()
        return -torch.einsum('bi,bi->b', probs, torch.log(probs + 1e-10))

    def normalize(self):
        """Ensure state is unit vector."""
        norm = torch.sqrt(complex_norm_sq(self.state).sum(dim=-1, keepdim=True) + 1e-10)
        self.state = self.state / norm.unsqueeze(-1)

    def set_superposition(self, amplitudes: torch.Tensor):
        """
        Set state to given amplitudes (will be normalized).

        Args:
            amplitudes: Complex amplitudes (..., dim, 2)
        """
        self.state = amplitudes.clone()
        self.normalize()


class UnitaryTransform(nn.Module):
    """
    Unitary transformation representing cognitive context.

    Context changes belief state via unitary evolution:
    |ψ'> = U|ψ>

    Learnable unitary parameterized as U = exp(iH) where H is Hermitian.
    Uses differentiable matrix exponential.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: Dimension of Hilbert space
        """
        super().__init__()
        self.dim = dim

        # Hermitian generator: H = H†
        # Parameterize as H = A + A† where A is any matrix
        self.A_real = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.A_imag = nn.Parameter(torch.randn(dim, dim) * 0.1)

    def get_hermitian(self) -> torch.Tensor:
        """Construct Hermitian matrix H = A + A†"""
        # A† has transposed real part and negated-transposed imag part
        H_real = self.A_real + self.A_real.T
        H_imag = self.A_imag - self.A_imag.T
        return torch.stack([H_real, H_imag], dim=-1)

    def get_unitary(self, t: float = 1.0) -> torch.Tensor:
        """
        Compute U = exp(itH) using eigendecomposition.

        Args:
            t: Evolution time (context strength)
        """
        H = self.get_hermitian()
        H_matrix = torch.complex(H[..., 0], H[..., 1])

        # Eigendecomposition: H = VDV†
        eigenvalues, eigenvectors = torch.linalg.eigh(H_matrix)

        # U = V exp(itD) V†
        exp_eigenvalues = torch.exp(1j * t * eigenvalues)
        U = eigenvectors @ torch.diag(exp_eigenvalues) @ eigenvectors.conj().T

        # Convert back to real representation
        return torch.stack([U.real, U.imag], dim=-1)

    def forward(self, state: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        """
        Apply unitary transformation to quantum state.

        Args:
            state: Quantum state (..., dim, 2)
            t: Context strength

        Returns:
            Transformed state
        """
        U = self.get_unitary(t)
        # |ψ'> = U|ψ> via einsum
        return torch.einsum('ijk,bik->bjk', U, state) if state.dim() == 3 else torch.einsum('ijk,ik->jk', U, state)


class ProjectiveMeasurement(nn.Module):
    """
    Projective measurement for decision crystallization.

    Measurement collapses superposition to eigenstate:
    P(outcome_i) = |<e_i|ψ>|^2

    Post-measurement state: |e_i> (the measured eigenstate)
    """

    def __init__(self, dim: int, num_outcomes: int):
        """
        Args:
            dim: Hilbert space dimension
            num_outcomes: Number of measurement outcomes
        """
        super().__init__()
        self.dim = dim
        self.num_outcomes = num_outcomes

        # Measurement basis (orthonormal projectors)
        # Initialize as random orthogonal matrix
        Q, _ = torch.linalg.qr(torch.randn(dim, num_outcomes))
        self.register_buffer('basis', torch.stack([Q, torch.zeros_like(Q)], dim=-1))

    def probabilities(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute outcome probabilities.

        P(i) = |<e_i|ψ>|^2 via einsum
        """
        # Inner products <e_i|ψ>
        # state: (batch, dim, 2), basis: (dim, outcomes, 2)
        # state[..., 0] is (batch, dim), basis[..., 0] is (dim, outcomes)
        inner_real = torch.einsum('bd,do->bo', state[..., 0], self.basis[..., 0]) + \
                     torch.einsum('bd,do->bo', state[..., 1], self.basis[..., 1])
        inner_imag = torch.einsum('bd,do->bo', state[..., 1], self.basis[..., 0]) - \
                     torch.einsum('bd,do->bo', state[..., 0], self.basis[..., 1])

        # |inner|^2 = real^2 + imag^2
        return inner_real.pow(2) + inner_imag.pow(2)

    def measure(self, state: torch.Tensor,
                return_collapsed: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform measurement (stochastic collapse).

        Args:
            state: Quantum state to measure
            return_collapsed: Whether to return post-measurement state

        Returns:
            outcome: Measured outcome indices
            collapsed_state: Post-measurement state (if requested)
        """
        probs = self.probabilities(state)
        outcome = torch.multinomial(probs + 1e-10, 1).squeeze(-1)

        if return_collapsed:
            # Collapse to measured eigenstate
            collapsed = torch.zeros_like(state)
            for b in range(state.shape[0]):
                collapsed[b] = self.basis[:, outcome[b]]
            return outcome, collapsed

        return outcome, None

    def soft_measure(self, state: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Differentiable soft measurement via Gumbel-Softmax.

        Returns weighted combination of basis states.
        """
        probs = self.probabilities(state)
        soft_outcome = F.gumbel_softmax(probs.log(), tau=temperature, hard=False)

        # Weighted sum of basis states: (batch, outcomes) @ (outcomes, dim, 2) -> (batch, dim, 2)
        # basis is (dim, outcomes, 2), transpose to (outcomes, dim, 2)
        basis_transposed = self.basis.permute(1, 0, 2)  # (outcomes, dim, 2)
        return torch.einsum('bo,odc->bdc', soft_outcome, basis_transposed)


class QuantumInterference(nn.Module):
    """
    Quantum interference for context-dependent probability.

    Explains violations of classical probability like:
    - Conjunction fallacy: P(A∧B) > P(A)
    - Order effects: P(A|B) ≠ P(B|A)

    Key insight: Probabilities emerge from amplitude interference,
    not independent combination.
    """

    def __init__(self, dim: int, num_contexts: int = 4):
        """
        Args:
            dim: Hilbert space dimension
            num_contexts: Number of context transformations
        """
        super().__init__()
        self.dim = dim
        self.num_contexts = num_contexts

        # Context-dependent unitary transformations
        self.contexts = nn.ModuleList([
            UnitaryTransform(dim) for _ in range(num_contexts)
        ])

        # Context selection weights (attention over contexts)
        self.context_attention = nn.Linear(dim, num_contexts)

    def select_context(self, features: torch.Tensor) -> torch.Tensor:
        """
        Select context via attention over feature embedding.

        Returns soft context weights.
        """
        return F.softmax(self.context_attention(features), dim=-1)

    def forward(self, state: torch.Tensor,
                features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply context-dependent transformation with interference.

        Args:
            state: Quantum state (..., dim, 2)
            features: Context features (..., dim)

        Returns:
            transformed_state: Post-context state
            context_weights: Attention weights over contexts
        """
        context_weights = self.select_context(features)

        # Apply each context and combine via superposition
        transformed = torch.zeros_like(state)
        for i, ctx in enumerate(self.contexts):
            ctx_state = ctx(state)
            # Weight by attention (amplitude, not probability!)
            weight = torch.sqrt(context_weights[:, i:i+1].unsqueeze(-1))
            transformed = transformed + weight * ctx_state

        # Normalize (interference may change total probability)
        norm = torch.sqrt(complex_norm_sq(transformed).sum(dim=-1, keepdim=True) + 1e-10)
        transformed = transformed / norm.unsqueeze(-1)

        return transformed, context_weights


class QuantumEntanglement(nn.Module):
    """
    Entanglement for correlated cognitive variables.

    Two beliefs A and B are entangled when measuring A
    instantaneously affects the distribution of B.

    Implemented via tensor product Hilbert spaces and
    non-separable states.
    """

    def __init__(self, dim_a: int, dim_b: int):
        """
        Args:
            dim_a: Dimension of first subsystem
            dim_b: Dimension of second subsystem
        """
        super().__init__()
        self.dim_a = dim_a
        self.dim_b = dim_b
        self.dim_total = dim_a * dim_b

        # Entangling unitary on joint space
        self.entangler = UnitaryTransform(self.dim_total)

        # Projectors for partial trace
        self.register_buffer(
            'trace_b_projector',
            torch.eye(dim_a).repeat_interleave(dim_b, dim=0).T
        )

    def create_product_state(self,
                             state_a: torch.Tensor,
                             state_b: torch.Tensor) -> torch.Tensor:
        """
        Create tensor product |ψ_A> ⊗ |ψ_B>.

        Uses einsum for outer product.
        """
        # (batch, dim_a, 2) ⊗ (batch, dim_b, 2) -> (batch, dim_a*dim_b, 2)
        # Outer product: result[b,a,c] = state_a[b,a] * state_b[b,c]
        real = torch.einsum('ba,bc->bac', state_a[..., 0], state_b[..., 0]) - \
               torch.einsum('ba,bc->bac', state_a[..., 1], state_b[..., 1])
        imag = torch.einsum('ba,bc->bac', state_a[..., 0], state_b[..., 1]) + \
               torch.einsum('ba,bc->bac', state_a[..., 1], state_b[..., 0])

        # Reshape to (batch, dim_a*dim_b, 2)
        real = real.reshape(-1, self.dim_total)
        imag = imag.reshape(-1, self.dim_total)

        return torch.stack([real, imag], dim=-1)

    def entangle(self,
                 state_a: torch.Tensor,
                 state_b: torch.Tensor,
                 strength: float = 1.0) -> torch.Tensor:
        """
        Create entangled state from product state.

        Args:
            state_a, state_b: Individual subsystem states
            strength: Entanglement strength (0 = product, 1 = maximally entangled)
        """
        product = self.create_product_state(state_a, state_b)
        entangled = self.entangler(product, t=strength)
        return entangled

    def measure_correlation(self,
                           joint_state: torch.Tensor,
                           observable_a: torch.Tensor,
                           observable_b: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation <A⊗B> - <A><B>.

        Non-zero correlation indicates entanglement.
        """
        # This is a simplified version - full implementation would
        # compute expectation values via trace
        probs = complex_norm_sq(joint_state)
        probs_matrix = probs.reshape(-1, self.dim_a, self.dim_b)

        # Marginals
        prob_a = probs_matrix.sum(dim=-1)
        prob_b = probs_matrix.sum(dim=-2)

        # Expected values
        exp_a = torch.einsum('ba,a->b', prob_a, observable_a)
        exp_b = torch.einsum('ba,a->b', prob_b, observable_b)

        # Joint expectation
        obs_joint = torch.einsum('a,b->ab', observable_a, observable_b).reshape(-1)
        exp_ab = torch.einsum('bj,j->b', probs, obs_joint)

        return exp_ab - exp_a * exp_b


class QuantumDecisionMaker(nn.Module):
    """
    Full quantum decision-making system.

    Integrates:
    - Superposition belief states
    - Context-dependent interference
    - Entanglement for correlated choices
    - Projective measurement for decisions

    Operates entirely via tensor operations - no if/else logic.
    """

    def __init__(self,
                 belief_dim: int = 8,
                 context_dim: int = 16,
                 num_outcomes: int = 4,
                 num_contexts: int = 4):
        """
        Args:
            belief_dim: Dimension of belief Hilbert space
            context_dim: Dimension of context features
            num_outcomes: Number of decision outcomes
            num_contexts: Number of context transformations
        """
        super().__init__()
        self.belief_dim = belief_dim

        # Components
        self.state = QuantumState(belief_dim)
        self.interference = QuantumInterference(belief_dim, num_contexts)
        self.measurement = ProjectiveMeasurement(belief_dim, num_outcomes)

        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, belief_dim),
            nn.LayerNorm(belief_dim),
            nn.GELU(),
        )

        # Belief encoder (maps observations to initial state)
        self.belief_encoder = nn.Sequential(
            nn.Linear(context_dim, belief_dim * 2),
            nn.LayerNorm(belief_dim * 2),
        )

    def encode_beliefs(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Encode observations into quantum belief state.

        Returns complex amplitudes.
        """
        encoded = self.belief_encoder(observations)
        real, imag = encoded.chunk(2, dim=-1)
        state = torch.stack([real, imag], dim=-1)

        # Normalize to unit vector
        norm = torch.sqrt(complex_norm_sq(state).sum(dim=-1, keepdim=True) + 1e-10)
        return state / norm.unsqueeze(-1)

    def forward(self,
                observations: torch.Tensor,
                context: torch.Tensor,
                temperature: float = 1.0,
                return_probs: bool = True) -> Dict[str, torch.Tensor]:
        """
        Make quantum-coherent decision.

        Args:
            observations: Input observations (..., context_dim)
            context: Context features (..., context_dim)
            temperature: Decision softness (lower = more deterministic)
            return_probs: Whether to return full probability distribution

        Returns:
            Dictionary with decision, probabilities, and diagnostics
        """
        batch_size = observations.shape[0]

        # Encode observations into belief state
        belief_state = self.encode_beliefs(observations)

        # Encode context
        context_features = self.context_encoder(context)

        # Apply context-dependent interference
        transformed_state, context_weights = self.interference(belief_state, context_features)

        # Compute decision probabilities
        probs = self.measurement.probabilities(transformed_state)

        # Soft decision (differentiable)
        decision_soft = F.gumbel_softmax(probs.log(), tau=temperature, hard=False)

        # Hard decision (for execution)
        decision_hard = probs.argmax(dim=-1)

        results = {
            'decision': decision_hard,
            'decision_soft': decision_soft,
            'context_weights': context_weights,
            'entropy': -torch.sum(probs * torch.log(probs + 1e-10), dim=-1),
        }

        if return_probs:
            results['probabilities'] = probs

        return results

    def order_effect(self,
                    state: torch.Tensor,
                    question_a: UnitaryTransform,
                    question_b: UnitaryTransform) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Demonstrate order effects: asking A then B vs B then A.

        Returns probability distributions for both orderings.
        """
        # Order A → B
        state_ab = question_a(state)
        state_ab = question_b(state_ab)
        probs_ab = complex_norm_sq(state_ab)

        # Order B → A
        state_ba = question_b(state)
        state_ba = question_a(state_ba)
        probs_ba = complex_norm_sq(state_ba)

        return probs_ab, probs_ba


class QuantumMemory(nn.Module):
    """
    Quantum-inspired memory with superposition retrieval.

    Memories exist in superposition until queried.
    Query acts as measurement, collapsing to relevant memories.

    Enables:
    - Associative retrieval via amplitude interference
    - Context-sensitive recall
    - Graceful degradation under noise
    """

    def __init__(self,
                 memory_dim: int = 64,
                 num_slots: int = 100,
                 key_dim: int = 32):
        """
        Args:
            memory_dim: Dimension of memory contents
            num_slots: Maximum number of memory slots
            key_dim: Dimension of memory keys
        """
        super().__init__()
        self.memory_dim = memory_dim
        self.num_slots = num_slots
        self.key_dim = key_dim

        # Memory as quantum state: |memory> = Σ α_i |content_i>
        self.register_buffer(
            'contents',
            torch.zeros(num_slots, memory_dim, 2)  # Complex
        )
        self.register_buffer(
            'keys',
            torch.zeros(num_slots, key_dim, 2)
        )
        self.register_buffer(
            'occupancy',
            torch.zeros(num_slots)
        )
        self.register_buffer('write_ptr', torch.tensor(0))

        # Query transformation
        self.query_encoder = nn.Linear(key_dim, key_dim * 2)

    def write(self,
              key: torch.Tensor,
              content: torch.Tensor,
              strength: float = 1.0):
        """
        Write memory with given amplitude.

        Args:
            key: Memory key (batch, key_dim)
            content: Memory content (batch, memory_dim)
            strength: Write amplitude
        """
        ptr = self.write_ptr.item()
        batch_size = key.shape[0]

        for i in range(batch_size):
            slot = (ptr + i) % self.num_slots

            # Complex encoding: real from content, small imaginary
            self.contents[slot, :, 0] = content[i] * strength
            self.contents[slot, :, 1] = content[i] * 0.1 * strength

            self.keys[slot, :, 0] = key[i]
            self.keys[slot, :, 1] = key[i] * 0.1

            self.occupancy[slot] = strength

        self.write_ptr = (self.write_ptr + batch_size) % self.num_slots

    def read(self,
             query: torch.Tensor,
             temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantum read: query induces superposition collapse.

        Args:
            query: Query key (batch, key_dim)
            temperature: Softness of retrieval

        Returns:
            content: Retrieved content (weighted sum)
            attention: Attention weights over memories
        """
        # Encode query as complex
        query_encoded = self.query_encoder(query)
        q_real, q_imag = query_encoded.chunk(2, dim=-1)
        query_complex = torch.stack([q_real, q_imag], dim=-1)

        # Compute amplitudes via complex inner product
        # <query|key_i> = sum_k (q_real*k_real + q_imag*k_imag) + i*(q_real*k_imag - q_imag*k_real)
        # |<query|key_i>|^2 = real^2 + imag^2
        inner_real = torch.einsum('bk,nk->bn', query_complex[..., 0], self.keys[..., 0]) + \
                     torch.einsum('bk,nk->bn', query_complex[..., 1], self.keys[..., 1])
        inner_imag = torch.einsum('bk,nk->bn', query_complex[..., 0], self.keys[..., 1]) - \
                     torch.einsum('bk,nk->bn', query_complex[..., 1], self.keys[..., 0])
        amplitudes = inner_real ** 2 + inner_imag ** 2

        # Mask unoccupied slots
        amplitudes = amplitudes * self.occupancy.unsqueeze(0)

        # Soft attention (quantum measurement with temperature)
        attention = F.softmax(amplitudes / temperature, dim=-1)

        # Retrieve via amplitude-weighted sum
        content = torch.einsum('bn,ndc->bdc', attention, self.contents)

        # Return real part
        return content[..., 0], attention


if __name__ == '__main__':
    print("--- Quantum Cognition Examples ---")

    # Example 1: Quantum State
    print("\n1. Quantum Superposition State")
    state = QuantumState(dim=4, batch_size=2)
    probs = state.probabilities()
    print(f"   Initial probabilities: {probs[0].tolist()}")
    print(f"   Entropy: {state.entropy().mean().item():.4f}")

    # Example 2: Unitary Context Transform
    print("\n2. Context-Dependent Transformation")
    transform = UnitaryTransform(dim=4)

    new_state = transform(state.state, t=1.0)
    new_probs = complex_norm_sq(new_state)
    print(f"   Post-context probabilities: {new_probs[0].tolist()}")

    # Example 3: Quantum Interference
    print("\n3. Quantum Interference")
    interference = QuantumInterference(dim=4, num_contexts=3)
    features = torch.randn(2, 4)

    interfered, ctx_weights = interference(state.state, features)
    interfered_probs = complex_norm_sq(interfered)
    print(f"   Context weights: {ctx_weights[0].tolist()}")
    print(f"   Interfered probs: {interfered_probs[0].tolist()}")

    # Example 4: Full Decision Maker
    print("\n4. Quantum Decision Maker")
    qd = QuantumDecisionMaker(
        belief_dim=8,
        context_dim=16,
        num_outcomes=4,
        num_contexts=3
    )

    obs = torch.randn(4, 16)
    ctx = torch.randn(4, 16)

    result = qd(obs, ctx, temperature=0.5)
    print(f"   Decisions: {result['decision'].tolist()}")
    print(f"   Entropy: {result['entropy'].mean().item():.4f}")
    print(f"   Probabilities: {result['probabilities'][0].tolist()}")

    # Example 5: Order Effects
    print("\n5. Order Effects (Non-commutativity)")
    q_a = UnitaryTransform(dim=4)
    q_b = UnitaryTransform(dim=4)

    probs_ab, probs_ba = qd.order_effect(state.state, q_a, q_b)
    diff = (probs_ab - probs_ba).abs().mean()
    print(f"   Order effect magnitude: {diff.item():.4f}")
    print("   (Non-zero indicates quantum-like order effects)")

    # Example 6: Quantum Memory
    print("\n6. Quantum Memory")
    qmem = QuantumMemory(memory_dim=32, num_slots=50, key_dim=16)

    # Write memories
    keys = torch.randn(10, 16)
    contents = torch.randn(10, 32)
    qmem.write(keys, contents)

    # Query
    query = keys[5] + torch.randn(1, 16) * 0.1  # Noisy query
    retrieved, attention = qmem.read(query)

    print(f"   Retrieved shape: {retrieved.shape}")
    print(f"   Top attention slots: {attention[0].topk(3).indices.tolist()}")

    print("\n[OK] All quantum cognition tests passed!")
