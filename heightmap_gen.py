import noise
from scipy.ndimage import gaussian_filter
import numpy as np
from opensimplex import OpenSimplex


def generate_landscape_heightmap(
    width, height, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None
):
    if seed is None:
        seed = np.random.randint(0, 1000000)
    noise_gen = OpenSimplex(seed=seed)

    def noise(nx, ny):
        # Rescale from -1.0:+1.0 to 0.0:1.0
        return noise_gen.noise2(nx, ny) / 2.0 + 0.5

    def octave_noise(nx, ny):
        value = 0.0
        amplitude = 1.0
        freq = 1.0
        for _ in range(octaves):
            value += noise(nx * freq, ny * freq) * amplitude
            amplitude *= persistence
            freq *= lacunarity
        return value

    heightmap = np.zeros((width, height))
    for y in range(height):
        for x in range(width):
            nx = x / scale - 0.5
            ny = y / scale - 0.5
            heightmap[y][x] = octave_noise(nx, ny)

    # Normalize the heightmap
    heightmap = (heightmap - np.min(heightmap)) / (
        np.max(heightmap) - np.min(heightmap)
    )

    # Apply some post-processing to enhance landscape features
    heightmap = np.power(heightmap, 1.2)
    heightmap = np.tanh(heightmap * 2) / 2 + 0.5

    return heightmap


def generate_landscape_heightmap_old(
    width, height, octaves=6, persistence=0.5, lacunarity=2.0, seed=None
):
    if seed is not None:
        np.random.seed(seed)

    def ridged_multifractal(x, y):
        """Generate ridged multifractal noise at (x,y)."""
        value = 0
        weight = 1.0
        for _ in range(octaves):
            signal = noise.pnoise2(
                x / scale,
                y / scale,
                octaves=1,
                persistence=0.5,
                lacunarity=2.0,
                repeatx=width,
                repeaty=height,
                base=np.random.randint(0, 1000),
            )
            signal = abs(signal)
            signal = 1.0 - signal
            signal *= signal
            signal *= weight
            value += signal
            weight = np.clip(signal * 2.0, 0, 1)
            scale *= lacunarity
        return value

    def fbm(x, y):
        """Generate fractal Brownian motion noise at (x,y)."""
        value = 0
        amplitude = 1.0
        frequency = 1.0
        for _ in range(octaves):
            value += amplitude * noise.pnoise2(
                x * frequency / scale,
                y * frequency / scale,
                octaves=1,
                persistence=0.5,
                lacunarity=2.0,
                repeatx=width,
                repeaty=height,
                base=np.random.randint(0, 1000),
            )
            amplitude *= persistence
            frequency *= lacunarity
        return value

    def domain_warp(x, y):
        """Apply domain warping to coordinates (x,y)."""
        qx = fbm(x + 5.2, y + 1.3)
        qy = fbm(x + 9.7, y + 2.8)
        return x + 4 * qx, y + 4 * qy

    heightmap = np.zeros((width, height))

    for i in range(width):
        for j in range(height):
            # Apply domain warping
            x, y = domain_warp(i, j)

            # Combine ridged multifractal and fbm
            heightmap[i][j] = ridged_multifractal(x, y) * 0.5 + fbm(x, y) * 0.5

    # Normalize
    heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())

    # Apply subtle Gaussian smoothing
    heightmap = gaussian_filter(heightmap, sigma=1)

    # Enhance terrain features
    heightmap = np.power(heightmap, 1.2)

    # Create more dramatic elevation changes
    heightmap = np.tanh(heightmap * 3) * 0.5 + 0.5

    # Final normalization
    heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())

    return heightmap
