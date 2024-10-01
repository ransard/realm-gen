import random
import noise
import numpy as np
from collections import deque
from sklearn.cluster import KMeans
import ollama
from realm_area import RealmArea
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
from collections import Counter

from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree


class RealmGenerator:
    def __init__(self, width, height, model_name: str = "llama3.1"):
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
        self.biome_colors = {
            0: (0, 0, 0.5),  # Deep Water: Dark Blue
            1: (0, 0, 1),  # Shallow Water: Blue
            2: (1, 1, 0.7),  # Beach: Pale Yellow
            3: (0.5, 0.8, 0.5),  # Plains: Light Green
            4: (0, 0.5, 0),  # Forest: Dark Green
            5: (0.5, 0.5, 0.5),  # Hills: Gray
            6: (0.7, 0.7, 0.7),  # Mountains: Light Gray
        }

    def generate_areas(self, biome_map, num_areas=5):
        # Flatten the 2D biome map into a list of (x, y, biome) tuples
        points = [
            (x, y, biome)
            for x in range(self.width)
            for y in range(self.height)
            for biome in [biome_map[x][y]]
        ]

        # Convert points to a numpy array for clustering
        X = np.array([(p[0], p[1], p[2]) for p in points])

        # Normalize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Adjust the weight of the biome feature
        biome_weight = (
            2.0  # Adjust this value to change the importance of biome in clustering
        )
        X_scaled[:, 2] *= biome_weight

        # Use KMeans for initial clustering
        kmeans = KMeans(n_clusters=num_areas, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        # Refine clusters based on local density and biome similarity
        refined_labels = self.refine_clusters(
            X, labels, radius=max(self.width, self.height) // 20
        )

        # Create a dictionary to store area information
        areas = {}

        # Process each cluster
        unique_labels = set(refined_labels)
        for label in unique_labels:
            # Get the points in this cluster
            cluster_points = [
                point for point, l in zip(points, refined_labels) if l == label
            ]

            # Calculate the bounding box for this area
            x_coords, y_coords, biomes = zip(*cluster_points)
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)

            # Get the most common biome in this area
            most_common_biome = Counter(biomes).most_common(1)[0][0]

            # Generate area name and description using Ollama
            area_name, area_description, area_characteristics = self.generate_area_info(
                self.biome_names[most_common_biome]
            )

            print(f"Generated area: {area_name}")
            print(f"Description: {area_description}")
            print(f"Characteristics: {area_characteristics}")

            # Store area information
            areas[(x1, y1, x2, y2)] = RealmArea(
                area_name, area_description, area_characteristics
            )

        self.save_image_with_areas(biome_map, areas, "areas.png")
        return areas

    def refine_clusters(self, X, labels, radius):
        tree = cKDTree(X)
        refined_labels = np.copy(labels)

        for i, point in enumerate(X):
            indices = tree.query_ball_point(point, radius)
            neighbor_labels = labels[indices]
            most_common = Counter(neighbor_labels).most_common(1)[0][0]
            refined_labels[i] = most_common

        return refined_labels

    def generate_areas_old(self, biome_map, num_areas=5):
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

        self.save_image_with_areas(biome_map, areas, "areas.png")
        return areas

    def generate_area_info(self, main_biome):
        print(f"Generating area info for {main_biome}...")
        prompt = f"""
        Generate a short, evocative name and a brief description for an area in a fantasy realm.
        The area is primarily composed of {main_biome.lower().replace('_', ' ')}.

        The characteristics should be a list of 2-5 key features like atmosphere, wildlife etc.
        The description should be 1-2 sentences and explain the uniqueness of the area. 
        The name should be 2-4 words long and should contain the essence of the area. 
        
        Format the response as follows:
        Characteristics: [Area Characteristic]
        Description: [Area Description]
        Name: [Area Name]
        """

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                # options={"temperature": 20.5, "seed": 115},
            )
            lines = response["response"].strip().split("\n")

            print(lines)

            name = (
                lines[2].split(": ", 1)[1]
                if lines[2].startswith("Name:")
                else "Unnamed Area"
            )
            description = (
                lines[1].split(": ", 1)[1]
                if len(lines) > 1 and lines[1].startswith("Description:")
                else "A mysterious area awaits exploration."
            )
            characteristics = (
                lines[0].split(": ", 1)[1].split(", ")
                if lines[0].startswith("Characteristics:")
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

        self.save_image(heightmap, "heightmap.png", cmap="terrain")
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

        self.save_image(final_heightmap, "heightmap.png", cmap="terrain")
        return final_heightmap

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

        self.save_image(biome_map, "biome_map.png", custom_cmap=self.biome_colors)
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

        self.save_image(rivers, "rivers.png", cmap="Blues")
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

        self.save_image_with_points(biome_map, villages, "villages.png", "Villages")
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

        self.save_image_with_points(
            biome_map,
            [(x, y) for x, y, _ in poi],
            "points_of_interest.png",
            "Points of Interest",
        )
        return poi

    def save_image(self, data, filename, cmap="viridis", custom_cmap=None):
        plt.figure(figsize=(10, 10))
        if custom_cmap:
            cmap = plt.matplotlib.colors.ListedColormap(list(custom_cmap.values()))
        plt.imshow(data, cmap=cmap)
        plt.colorbar()
        plt.title(filename.split(".")[0].replace("_", " ").title())
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def save_image_with_points(self, background, points, filename, title):
        plt.figure(figsize=(10, 10))
        plt.imshow(
            background,
            cmap=plt.matplotlib.colors.ListedColormap(list(self.biome_colors.values())),
        )
        x, y = zip(*points)
        plt.scatter(y, x, c="red", s=20)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def save_image_with_areas(self, background, areas, filename):
        plt.figure(figsize=(10, 10))
        plt.imshow(
            background,
            cmap=plt.matplotlib.colors.ListedColormap(list(self.biome_colors.values())),
        )
        for (x1, y1, x2, y2), area_info in areas.items():
            rect = plt.Rectangle(
                (y1, x1), y2 - y1, x2 - x1, fill=False, edgecolor="red"
            )

            print(f"Area: {area_info}")

            plt.gca().add_patch(rect)
            plt.text(
                (y1 + y2) / 2,
                (x1 + x2) / 2,
                area_info.name,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.7),
            )
        plt.title("Realm Areas")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def generate_realm(self):
        print("Generating realm...")
        heightmap = self.generate_improved_heightmap(seed=42)
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
