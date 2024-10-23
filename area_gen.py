import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from realm_area import RealmArea
from collections import Counter
from scipy.spatial import cKDTree
import ollama

from constants import climate_names, model_name


def refine_clusters(X, labels, radius):
    tree = cKDTree(X)
    refined_labels = np.copy(labels)

    for i, point in enumerate(X):
        indices = tree.query_ball_point(point, radius)
        neighbor_labels = labels[indices]
        most_common = Counter(neighbor_labels).most_common(1)[0][0]
        refined_labels[i] = most_common

    return refined_labels


def generate_areas(width, height, biome_map, num_areas=5):
    # Flatten the 2D biome map into a list of (x, y, biome) tuples
    points = [
        (x, y, biome)
        for x in range(width)
        for y in range(height)
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
    refined_labels = refine_clusters(X, labels, radius=max(width, height) // 20)

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
        area_name, area_description, area_characteristics = generate_area_info(
            climate_names[most_common_biome]
        )

        print(f"Generated area: {area_name}")
        print(f"Description: {area_description}")
        print(f"Characteristics: {area_characteristics}")

        # Store area information
        areas[(x1, y1, x2, y2)] = RealmArea(
            area_name, area_description, area_characteristics
        )

    return areas


def generate_area_info(main_biome):
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
            model=model_name,
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
