import random
import numpy as np
from collections import deque
from sklearn.cluster import DBSCAN
from collections import Counter

from image_handler import ImageHandler
from heightmap_gen import generate_landscape_heightmap
from climate_gen import apply_biomes
from area_gen import generate_areas
from constants import climate_colors


class RealmGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.image_handler = ImageHandler(climate_colors)

    def generate_rivers(self, heightmap, num_rivers=5, max_length=00, min_length=10):
        rivers = np.zeros((self.width, self.height), dtype=bool)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for _ in range(num_rivers):
            # Find a starting point at higher elevation
            start_x, start_y = self.find_river_source(heightmap)
            if start_x is None or start_y is None:
                continue  # Skip this river if no suitable source is found

            river = [(start_x, start_y)]
            visited = set([(start_x, start_y)])

            x, y = start_x, start_y
            for _ in range(max_length):
                rivers[x, y] = True

                # Find the lowest neighboring point
                neighbors = []
                for dx, dy in directions:
                    nx, ny = (x + dx) % self.width, (y + dy) % self.height
                    if (nx, ny) not in visited:
                        neighbors.append((heightmap[nx, ny], nx, ny))

                if not neighbors:
                    break  # River is blocked

                # Move to the lowest neighbor
                _, next_x, next_y = min(neighbors)

                # Stop if we've reached water or can't flow downhill anymore
                if (
                    heightmap[next_x, next_y] < -0.2
                    or heightmap[next_x, next_y] >= heightmap[x, y]
                ):
                    break

                river.append((next_x, next_y))
                visited.add((next_x, next_y))
                x, y = next_x, next_y

            # Only keep rivers that reach a certain length
            if len(river) > min_length:
                for rx, ry in river:
                    rivers[rx, ry] = True

        self.image_handler.save_image(rivers, "rivers.png", cmap="Blues")
        return rivers

    def find_river_source(self, heightmap, attempts=0, elevation_threshold=0.6):
        height_sorted = sorted(
            [
                (heightmap[x, y], x, y)
                for x in range(self.width)
                for y in range(self.height)
            ],
            reverse=True,
        )

        for _ in range(attempts):
            # Randomly select from the top % of elevation points
            _, x, y = random.choice(height_sorted[: int(len(height_sorted) * 0.1)])
            if heightmap[x, y] > elevation_threshold:
                return x, y

        # If we couldn't find a point above the threshold, return the highest point
        _, x, y = height_sorted[0]
        return x, y

    def place_villages(self, biome_map, rivers, num_villages=10):
        villages = []
        for _ in range(num_villages):
            attempts = 0
            while attempts < 100:  # Limit attempts to avoid infinite loop
                x, y = random.randint(0, self.width - 1), random.randint(
                    0, self.height - 1
                )
                # if biome_map[x][y] in [
                #     biomes["PLAINS"],
                #     biomes["FOREST"],
                # ] and self.is_near_water(x, y, rivers):
                villages.append((x, y))
                break
                # attempts += 1

        if len(villages) > 0:
            self.image_handler.save_image_with_points(
                biome_map, villages, "villages.png", "Villages"
            )
        return villages

    def is_near_water(self, x, y, rivers, distance=5):
        for dx in range(-distance, distance + 1):
            for dy in range(-distance, distance + 1):
                nx, ny = (x + dx) % self.width, (y + dy) % self.height
                if rivers[nx][ny]:
                    return True
        return False

    def place_poi(self, biome_map, num_poi=20):
        poi = []
        poi_types = ["DUNGEON", "RUINS", "MAGICAL_SITE", "NATURAL_WONDER"]
        for _ in range(num_poi):
            poi_type = random.choice(poi_types)
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            if poi_type == "DUNGEON" and biome_map[x][y] in [
                "Alpine",
                "Tundra",
                "Polar",
            ]:
                poi.append((x, y, poi_type))
            elif poi_type == "RUINS" and biome_map[x][y] in [
                "Tropical Rainforest",
                "Mediterranean",
                "Temperate Forest",
                "Continental",
            ]:
                poi.append((x, y, poi_type))
            elif poi_type == "MAGICAL_SITE":  # Can be anywhere
                poi.append((x, y, poi_type))
            elif poi_type == "NATURAL_WONDER" and biome_map[x][y] != "Deep Water":
                poi.append((x, y, poi_type))

        self.image_handler.save_image_with_points(
            biome_map,
            [(x, y) for x, y, _ in poi],
            "points_of_interest.png",
            "Points of Interest",
        )
        return poi

    def generate_realm(self):
        print("Generating realm...")
        heightmap = generate_landscape_heightmap(self.width, self.height)
        self.image_handler.save_image(heightmap, "heightmap.png")
        latitudes = np.linspace(0, 90, self.height)
        print("Generated heightmap")
        biome_map = apply_biomes(heightmap, latitudes)
        self.image_handler.save_image(
            biome_map, "biome_map.png", custom_cmap=climate_colors
        )
        print("Applied biomes")
        rivers = self.generate_rivers(heightmap)
        print("Generated rivers")
        villages = self.place_villages(biome_map, rivers)
        print("Placed villages")
        points_of_interest = self.place_poi(biome_map)
        print("Placed points of interest")
        areas = generate_areas(self.width, self.height, biome_map)
        self.image_handler.save_image_with_areas(biome_map, areas, "areas.png")
        print("Generated areas")

        return {
            "heightmap": heightmap,
            "biome_map": biome_map,
            "rivers": rivers,
            "villages": villages,
            "points_of_interest": points_of_interest,
            "areas": areas,
        }
