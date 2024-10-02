import random
import noise
from scipy.ndimage import gaussian_filter
import numpy as np


class HeightmapGenerator:
    def __init__(self, width, height, image_handler):
        self.width = width
        self.height = height
        self.image_handler = image_handler

    def generate_heightmap(
        self, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0
    ):
        heightmap = np.zeros((self.width, self.height))
        for i in range(self.width):
            for j in range(self.height):
                heightmap[i][j] = noise.pnoise2(
                    i / scale,
                    j / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=self.width,
                    repeaty=self.height,
                    base=random.randint(0, 1000),
                )

        self.image_handler.save_image(heightmap, "heightmap.png", cmap="terrain")
        return heightmap

    def generate_improved_heightmap(
        self,
        scale=100.0,
        octaves=6,
        persistence=0.5,
        lacunarity=2.0,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)

        width = self.width
        height = self.height

        heightmap = np.zeros((width, height))

        # Generate base noise
        for i in range(width):
            for j in range(height):
                heightmap[i][j] = noise.pnoise2(
                    i / scale,
                    j / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=width,
                    repeaty=height,
                    base=np.random.randint(0, 1000),
                )

        # Normalize the heightmap
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())

        # Apply Gaussian smoothing
        smoothed_heightmap = gaussian_filter(heightmap, sigma=2)

        # Enhance contrast
        enhanced_heightmap = np.power(smoothed_heightmap, 1.5)

        # Renormalize
        final_heightmap = (enhanced_heightmap - enhanced_heightmap.min()) / (
            enhanced_heightmap.max() - enhanced_heightmap.min()
        )

        self.image_handler.save_image(final_heightmap, "heightmap.png", cmap="terrain")
        return final_heightmap
