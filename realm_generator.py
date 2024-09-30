import random
import noise
import numpy as np
from collections import deque
from sklearn.cluster import KMeans
import ollama
from realm_area import RealmArea


class RealmGenerator:
    def __init__(self, width, height, model_name: str = "qwen2.5:14b"):
        self.width = width
        self.height = height
        self.model_name = model_name
        self.biomes = {
            "DEEP_WATER": 0,
            "SHALLOW_WATER": 1,
            "BEACH": 2,
            "PLAINS": 3,
            "FOREST": 4,
            "HILLS": 5,
            "MOUNTAINS": 6,
        }
        self.biome_names = {v: k for k, v in self.biomes.items()}

    def generate_areas(self, biome_map, num_areas=5):
        # Flatten the 2D biome map into a list of (x, y, biome) tuples
        points = [
            (x, y, biome)
            for x in range(self.width)
            for y in range(self.height)
            for biome in [biome_map[x][y]]
        ]

        # Use KMeans clustering to identify distinct areas
        kmeans = KMeans(n_clusters=num_areas, random_state=42)
        kmeans.fit(points)

        # Create a dictionary to store area information
        areas = {}
        for i in range(num_areas):
            # Get the points in this cluster
            cluster_points = [
                point for point, label in zip(points, kmeans.labels_) if label == i
            ]

            # Calculate the bounding box for this area
            x_coords, y_coords, biomes = zip(*cluster_points)
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)

            # Get the most common biome in this area
            most_common_biome = max(set(biomes), key=biomes.count)

            # Generate area name and description using Ollama
            area_name, area_description, area_characteristics = self.generate_area_info(
                self.biome_names[most_common_biome]
            )

            print(f"Generated area: {area_name})")
            print(f"Description: {area_description}")
            print(f"Characteristics: {area_characteristics}")

            # Store area information
            areas[(x1, y1, x2, y2)] = RealmArea(
                area_name, area_description, area_characteristics
            )
            # {
            #     "name": area_name,
            #     "description": area_description,
            #     "main_biome": self.biome_names[most_common_biome],
            # }

        return areas

    def generate_area_info(self, main_biome):
        prompt = f"""
        Generate a short, evocative name and a brief description for an area in a fantasy realm.
        The area is primarily composed of {main_biome.lower().replace('_', ' ')}.
        
        Format the response as follows:
        Name: [Area Name]
        Description: [Area Description]
        Characteristics: [Area Characteristic]

        The name should be 2-4 words long. The description should be 1-2 sentences. The characteristics should be a list of 2-5 key features like atmosphere, wildlife etc.
        """

        try:
            response = ollama.generate(model=self.model_name, prompt=prompt)
            lines = response["response"].strip().split("\n")
            name = (
                lines[0].split(": ", 1)[1]
                if lines[0].startswith("Name:")
                else "Unnamed Area"
            )
            description = (
                lines[1].split(": ", 1)[1]
                if len(lines) > 1 and lines[1].startswith("Description:")
                else "A mysterious area awaits exploration."
            )
            characteristics = (
                lines[2].split(": ", 1)[1].split(", ")
                if len(lines) > 2 and lines[2].startswith("Characteristics:")
                else "N/A"
            )

            return name, description, characteristics
        except Exception as e:
            print(f"Error generating area info: {str(e)}")
            return "Unnamed Area", "A mysterious area awaits exploration."

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
        return heightmap

    def apply_biomes(self, heightmap):
        biome_map = np.zeros((self.width, self.height), dtype=int)
        for i in range(self.width):
            for j in range(self.height):
                height = heightmap[i][j]
                if height < -0.2:
                    biome_map[i][j] = self.biomes["DEEP_WATER"]
                elif height < 0:
                    biome_map[i][j] = self.biomes["SHALLOW_WATER"]
                elif height < 0.1:
                    biome_map[i][j] = self.biomes["BEACH"]
                elif height < 0.3:
                    biome_map[i][j] = self.biomes["PLAINS"]
                elif height < 0.5:
                    biome_map[i][j] = self.biomes["FOREST"]
                elif height < 0.7:
                    biome_map[i][j] = self.biomes["HILLS"]
                else:
                    biome_map[i][j] = self.biomes["MOUNTAINS"]
        return biome_map

    def generate_rivers(self, heightmap, num_rivers=5, max_length=1000, min_length=10):
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

        return rivers

    def find_river_source(self, heightmap, attempts=100, elevation_threshold=0.6):
        height_sorted = sorted(
            [
                (heightmap[x, y], x, y)
                for x in range(self.width)
                for y in range(self.height)
            ],
            reverse=True,
        )

        for _ in range(attempts):
            # Randomly select from the top 10% of elevation points
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
                if biome_map[x][y] in [
                    self.biomes["PLAINS"],
                    self.biomes["FOREST"],
                ] and self.is_near_water(x, y, rivers):
                    villages.append((x, y))
                    break
                attempts += 1
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
                self.biomes["HILLS"],
                self.biomes["MOUNTAINS"],
            ]:
                poi.append((x, y, poi_type))
            elif poi_type == "RUINS" and biome_map[x][y] in [
                self.biomes["PLAINS"],
                self.biomes["FOREST"],
            ]:
                poi.append((x, y, poi_type))
            elif poi_type == "MAGICAL_SITE":  # Can be anywhere
                poi.append((x, y, poi_type))
            elif (
                poi_type == "NATURAL_WONDER"
                and biome_map[x][y] != self.biomes["DEEP_WATER"]
            ):
                poi.append((x, y, poi_type))
        return poi

    def generate_realm(self):
        print("Generating realm...")
        heightmap = self.generate_heightmap()
        print("Generated heightmap")
        biome_map = self.apply_biomes(heightmap)
        print("Applied biomes")
        rivers = self.generate_rivers(heightmap)
        print("Generated rivers")
        villages = self.place_villages(biome_map, rivers)
        print("Placed villages")
        points_of_interest = self.place_poi(biome_map)
        print("Placed points of interest")
        areas = self.generate_areas(biome_map)
        print("Generated areas")

        return {
            "heightmap": heightmap,
            "biome_map": biome_map,
            "rivers": rivers,
            "villages": villages,
            "points_of_interest": points_of_interest,
            "areas": areas,
        }
