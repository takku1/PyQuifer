
import torch
import torch.nn as nn


class PerlinNoise(nn.Module):
    """
    Pure PyTorch implementation of Perlin noise (GPU-compatible, differentiable).
    Supports 2D, 3D, and 4D noise generation.
    """

    def __init__(self, seed: int = 0):
        super().__init__()
        self.seed = seed
        # Generate permutation table
        torch.manual_seed(seed)
        perm = torch.randperm(256)
        # Double the permutation table to avoid overflow
        perm = torch.cat([perm, perm])
        self.register_buffer('perm', perm)

        # Gradients for 2D (8 directions)
        grad2 = torch.tensor([
            [1, 1], [-1, 1], [1, -1], [-1, -1],
            [1, 0], [-1, 0], [0, 1], [0, -1]
        ], dtype=torch.float32)
        self.register_buffer('grad2', grad2)

        # Gradients for 3D (12 directions)
        grad3 = torch.tensor([
            [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
            [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
            [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]
        ], dtype=torch.float32)
        self.register_buffer('grad3', grad3)

        # Gradients for 4D (32 directions)
        grad4 = torch.tensor([
            [0, 1, 1, 1], [0, 1, 1, -1], [0, 1, -1, 1], [0, 1, -1, -1],
            [0, -1, 1, 1], [0, -1, 1, -1], [0, -1, -1, 1], [0, -1, -1, -1],
            [1, 0, 1, 1], [1, 0, 1, -1], [1, 0, -1, 1], [1, 0, -1, -1],
            [-1, 0, 1, 1], [-1, 0, 1, -1], [-1, 0, -1, 1], [-1, 0, -1, -1],
            [1, 1, 0, 1], [1, 1, 0, -1], [1, -1, 0, 1], [1, -1, 0, -1],
            [-1, 1, 0, 1], [-1, 1, 0, -1], [-1, -1, 0, 1], [-1, -1, 0, -1],
            [1, 1, 1, 0], [1, 1, -1, 0], [1, -1, 1, 0], [1, -1, -1, 0],
            [-1, 1, 1, 0], [-1, 1, -1, 0], [-1, -1, 1, 0], [-1, -1, -1, 0]
        ], dtype=torch.float32)
        self.register_buffer('grad4', grad4)

    def _fade(self, t: torch.Tensor) -> torch.Tensor:
        """Quintic interpolation curve: 6t^5 - 15t^4 + 10t^3"""
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Linear interpolation"""
        return a + t * (b - a)

    def _hash(self, *indices) -> torch.Tensor:
        """Hash function using permutation table"""
        result = torch.zeros_like(indices[0])
        for idx in indices:
            result = self.perm[(result.long() + idx.long()) % 256]
        return result

    def noise2d(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate 2D Perlin noise"""
        # Integer coordinates
        xi = x.floor().long() & 255
        yi = y.floor().long() & 255

        # Fractional coordinates
        xf = x - x.floor()
        yf = y - y.floor()

        # Fade curves
        u = self._fade(xf)
        v = self._fade(yf)

        # Hash coordinates of the 4 corners
        aa = self._hash(xi, yi) % 8
        ab = self._hash(xi, yi + 1) % 8
        ba = self._hash(xi + 1, yi) % 8
        bb = self._hash(xi + 1, yi + 1) % 8

        # Gradient dot products
        g_aa = (self.grad2[aa] * torch.stack([xf, yf], dim=-1)).sum(dim=-1)
        g_ab = (self.grad2[ab] * torch.stack([xf, yf - 1], dim=-1)).sum(dim=-1)
        g_ba = (self.grad2[ba] * torch.stack([xf - 1, yf], dim=-1)).sum(dim=-1)
        g_bb = (self.grad2[bb] * torch.stack([xf - 1, yf - 1], dim=-1)).sum(dim=-1)

        # Interpolation
        x1 = self._lerp(g_aa, g_ba, u)
        x2 = self._lerp(g_ab, g_bb, u)
        return self._lerp(x1, x2, v)

    def noise3d(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Generate 3D Perlin noise"""
        # Integer coordinates
        xi = x.floor().long() & 255
        yi = y.floor().long() & 255
        zi = z.floor().long() & 255

        # Fractional coordinates
        xf = x - x.floor()
        yf = y - y.floor()
        zf = z - z.floor()

        # Fade curves
        u = self._fade(xf)
        v = self._fade(yf)
        w = self._fade(zf)

        # Hash coordinates of the 8 corners
        aaa = self._hash(xi, yi, zi) % 12
        aab = self._hash(xi, yi, zi + 1) % 12
        aba = self._hash(xi, yi + 1, zi) % 12
        abb = self._hash(xi, yi + 1, zi + 1) % 12
        baa = self._hash(xi + 1, yi, zi) % 12
        bab = self._hash(xi + 1, yi, zi + 1) % 12
        bba = self._hash(xi + 1, yi + 1, zi) % 12
        bbb = self._hash(xi + 1, yi + 1, zi + 1) % 12

        # Gradient dot products
        def dot3(grad_idx, dx, dy, dz):
            g = self.grad3[grad_idx]
            return g[..., 0] * dx + g[..., 1] * dy + g[..., 2] * dz

        g_aaa = dot3(aaa, xf, yf, zf)
        g_aab = dot3(aab, xf, yf, zf - 1)
        g_aba = dot3(aba, xf, yf - 1, zf)
        g_abb = dot3(abb, xf, yf - 1, zf - 1)
        g_baa = dot3(baa, xf - 1, yf, zf)
        g_bab = dot3(bab, xf - 1, yf, zf - 1)
        g_bba = dot3(bba, xf - 1, yf - 1, zf)
        g_bbb = dot3(bbb, xf - 1, yf - 1, zf - 1)

        # Trilinear interpolation
        x1 = self._lerp(g_aaa, g_baa, u)
        x2 = self._lerp(g_aba, g_bba, u)
        x3 = self._lerp(g_aab, g_bab, u)
        x4 = self._lerp(g_abb, g_bbb, u)
        y1 = self._lerp(x1, x2, v)
        y2 = self._lerp(x3, x4, v)
        return self._lerp(y1, y2, w)

    def noise4d(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Generate 4D Perlin noise"""
        # Integer coordinates
        xi = x.floor().long() & 255
        yi = y.floor().long() & 255
        zi = z.floor().long() & 255
        wi = w.floor().long() & 255

        # Fractional coordinates
        xf = x - x.floor()
        yf = y - y.floor()
        zf = z - z.floor()
        wf = w - w.floor()

        # Fade curves
        u = self._fade(xf)
        v = self._fade(yf)
        s = self._fade(zf)
        t = self._fade(wf)

        # Hash all 16 corners
        def h(dx, dy, dz, dw):
            return self._hash(xi + dx, yi + dy, zi + dz, wi + dw) % 32

        # Gradient dot products
        def dot4(grad_idx, dx, dy, dz, dw):
            g = self.grad4[grad_idx]
            return g[..., 0] * dx + g[..., 1] * dy + g[..., 2] * dz + g[..., 3] * dw

        # All 16 corners (4D hypercube)
        g0000 = dot4(h(0, 0, 0, 0), xf, yf, zf, wf)
        g0001 = dot4(h(0, 0, 0, 1), xf, yf, zf, wf - 1)
        g0010 = dot4(h(0, 0, 1, 0), xf, yf, zf - 1, wf)
        g0011 = dot4(h(0, 0, 1, 1), xf, yf, zf - 1, wf - 1)
        g0100 = dot4(h(0, 1, 0, 0), xf, yf - 1, zf, wf)
        g0101 = dot4(h(0, 1, 0, 1), xf, yf - 1, zf, wf - 1)
        g0110 = dot4(h(0, 1, 1, 0), xf, yf - 1, zf - 1, wf)
        g0111 = dot4(h(0, 1, 1, 1), xf, yf - 1, zf - 1, wf - 1)
        g1000 = dot4(h(1, 0, 0, 0), xf - 1, yf, zf, wf)
        g1001 = dot4(h(1, 0, 0, 1), xf - 1, yf, zf, wf - 1)
        g1010 = dot4(h(1, 0, 1, 0), xf - 1, yf, zf - 1, wf)
        g1011 = dot4(h(1, 0, 1, 1), xf - 1, yf, zf - 1, wf - 1)
        g1100 = dot4(h(1, 1, 0, 0), xf - 1, yf - 1, zf, wf)
        g1101 = dot4(h(1, 1, 0, 1), xf - 1, yf - 1, zf, wf - 1)
        g1110 = dot4(h(1, 1, 1, 0), xf - 1, yf - 1, zf - 1, wf)
        g1111 = dot4(h(1, 1, 1, 1), xf - 1, yf - 1, zf - 1, wf - 1)

        # 4D interpolation
        x00 = self._lerp(g0000, g1000, u)
        x01 = self._lerp(g0001, g1001, u)
        x10 = self._lerp(g0010, g1010, u)
        x11 = self._lerp(g0011, g1011, u)
        x20 = self._lerp(g0100, g1100, u)
        x21 = self._lerp(g0101, g1101, u)
        x30 = self._lerp(g0110, g1110, u)
        x31 = self._lerp(g0111, g1111, u)

        y0 = self._lerp(x00, x20, v)
        y1 = self._lerp(x01, x21, v)
        y2 = self._lerp(x10, x30, v)
        y3 = self._lerp(x11, x31, v)

        z0 = self._lerp(y0, y2, s)
        z1 = self._lerp(y1, y3, s)

        return self._lerp(z0, z1, t)


class PerturbationLayer(nn.Module):
    """
    Generates N-dimensional Perlin noise as a PyTorch tensor.
    This layer acts as the "Perturbation Layer" or "Canvas" for the generative model.

    Pure PyTorch implementation - GPU-compatible and differentiable.
    Supports 2D, 3D, and 4D noise with fractal octaves (fBm).
    """

    def __init__(self, dim: int, scale: float = 10.0, octaves: int = 6,
                 persistence: float = 0.5, lacunarity: float = 2.0, seed: int = None):
        super().__init__()
        if dim < 2:
            raise ValueError("PerturbationLayer requires at least 2 dimensions.")
        # For dim > 4, we'll use smoothed random noise as a computationally efficient fallback
        self._use_perlin = (2 <= dim <= 4)
        self.dim = dim
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = seed if seed is not None else torch.randint(0, 2**31, (1,)).item()

        if self._use_perlin:
            self.noise = PerlinNoise(seed=self.seed)
        else:
            self.noise = None  # Will use smoothed random noise

    def forward(self, shape: tuple, time_offset: float = 0.0) -> torch.Tensor:
        """
        Generates noise for a given shape using fractal Brownian motion (fBm) for 2-4D,
        or smoothed random noise for higher dimensions.

        Args:
            shape (tuple): The shape of the output noise tensor.
            time_offset (float): An offset for the 4th dimension (time) for 4D noise.

        Returns:
            torch.Tensor: A tensor containing noise, normalized to [0, 1].
        """
        if len(shape) != self.dim:
            raise ValueError(f"Expected shape of length {self.dim} for {self.dim}D noise, but got {len(shape)}.")

        # For high dimensions (>4), use smoothed random noise
        if not self._use_perlin:
            return self._generate_smoothed_random_noise(shape)

        # Get the device from the noise module's buffers
        device = self.noise.perm.device

        # Create coordinate grids
        ranges = [torch.linspace(0, s - 1, s, device=device) for s in shape]
        grids = torch.meshgrid(*ranges, indexing='ij')

        # Generate fractal noise (fBm)
        noise_tensor = torch.zeros(shape, device=device)
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0

        for _ in range(self.octaves):
            if self.dim == 2:
                x = grids[0] * frequency / self.scale
                y = grids[1] * frequency / self.scale
                noise_tensor += self.noise.noise2d(x, y) * amplitude
            elif self.dim == 3:
                x = grids[0] * frequency / self.scale
                y = grids[1] * frequency / self.scale
                z = grids[2] * frequency / self.scale
                noise_tensor += self.noise.noise3d(x, y, z) * amplitude
            elif self.dim == 4:
                x = grids[0] * frequency / self.scale
                y = grids[1] * frequency / self.scale
                z = grids[2] * frequency / self.scale
                w = (grids[3] + time_offset) * frequency / self.scale
                noise_tensor += self.noise.noise4d(x, y, z, w) * amplitude

            max_value += amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity

        # Normalize to [-1, 1] first, then to [0, 1]
        noise_tensor = noise_tensor / max_value

        # Normalize to [0, 1]
        noise_min = noise_tensor.min()
        noise_max = noise_tensor.max()
        if noise_max == noise_min:
            return torch.zeros_like(noise_tensor)
        normalized_noise = (noise_tensor - noise_min) / (noise_max - noise_min)
        return normalized_noise

    def _generate_smoothed_random_noise(self, shape: tuple) -> torch.Tensor:
        """
        Generate fractal random noise for high-dimensional cases (dim > 4).
        Uses multi-octave random noise summation - efficient for any dimension.
        For creative jitter purposes, this provides the right statistical properties
        without needing true spatial correlation.
        """
        torch.manual_seed(self.seed)

        noise_tensor = torch.zeros(shape)
        amplitude = 1.0
        max_value = 0.0

        for octave in range(self.octaves):
            # Each octave uses a different random seed offset for variety
            torch.manual_seed(self.seed + octave * 1000)
            raw_noise = torch.randn(shape)
            noise_tensor += raw_noise * amplitude
            max_value += amplitude
            amplitude *= self.persistence

        # Normalize to [0, 1]
        noise_tensor = noise_tensor / max_value
        noise_min = noise_tensor.min()
        noise_max = noise_tensor.max()
        if noise_max == noise_min:
            return torch.zeros_like(noise_tensor)
        return (noise_tensor - noise_min) / (noise_max - noise_min)


if __name__ == '__main__':
    print("--- PerturbationLayer 3D Example ---")
    perturb_3d = PerturbationLayer(dim=3, scale=20.0, octaves=4)
    shape_3d = (32, 32, 32)
    noise_3d = perturb_3d(shape_3d)
    print(f"3D Noise shape: {noise_3d.shape}")
    print(f"3D Noise min: {noise_3d.min().item():.4f}, max: {noise_3d.max().item():.4f}")

    print("\n--- PerturbationLayer 4D Example ---")
    perturb_4d = PerturbationLayer(dim=4, scale=15.0, octaves=3)
    shape_4d = (16, 16, 16, 8)
    noise_4d = perturb_4d(shape_4d)
    print(f"4D Noise shape: {noise_4d.shape}")
    print(f"4D Noise min: {noise_4d.min().item():.4f}, max: {noise_4d.max().item():.4f}")

    print("\n--- PerturbationLayer 2D Example ---")
    perturb_2d = PerturbationLayer(dim=2, scale=10.0, octaves=5)
    shape_2d = (64, 64)
    noise_2d = perturb_2d(shape_2d)
    print(f"2D Noise shape: {noise_2d.shape}")
    print(f"2D Noise min: {noise_2d.min().item():.4f}, max: {noise_2d.max().item():.4f}")

    # Test GPU support if available
    if torch.cuda.is_available():
        print("\n--- GPU Test ---")
        perturb_gpu = PerturbationLayer(dim=3, scale=10.0, octaves=4).cuda()
        noise_gpu = perturb_gpu((32, 32, 32))
        print(f"GPU Noise device: {noise_gpu.device}")
        print(f"GPU Noise shape: {noise_gpu.shape}")
