import numpy as np
import matplotlib.pyplot as plt
from constants import climate_colors, climates

# Constants for climate classification
TROPICAL_LIMIT = 23  # Tropical region roughly between 0° and 23° latitude
SUBTROPICAL_LIMIT = 30  # Deserts often found near 30° latitude
TEMPERATE_LIMIT = 60  # Temperate climates between 30° and 60° latitude
ALTITUDE_TUNDRA_THRESHOLD = 1500  # Altitude above which we get tundra
ALTITUDE_ALPINE_THRESHOLD = 3000  # Altitude for alpine climates
MAX_ALTITUDE = 4000


"""
Module to generate and visualize climate maps based on latitude and altitude.

Functions:
- determine_climate: Determine the climate based on latitude and altitude.
- apply_biomes: Apply biomes to a heightmap based on latitude and altitude.
- visualize_climate_map: Visualize the climate map using matplotlib.
"""


# Function to determine the climate based on latitude and altitude
def apply_biomes(heightmap, latitudes):
    rows, cols = heightmap.shape
    climate_map = np.zeros_like(heightmap, dtype=int)

    for i in range(rows):
        for j in range(cols):
            altitude = heightmap[i, j] * MAX_ALTITUDE
            latitude = latitudes[i]

            # Determine the climate based on latitude and altitude
            if latitude < TROPICAL_LIMIT:
                if altitude > ALTITUDE_ALPINE_THRESHOLD:
                    climate_map[i, j] = climates["ALPINE"]
                elif altitude > ALTITUDE_TUNDRA_THRESHOLD:
                    climate_map[i, j] = climates["TUNDRA"]
                elif 10 < latitude < TROPICAL_LIMIT:
                    climate_map[i, j] = climates["SAVANNA"]
                else:
                    climate_map[i, j] = climates["TROPICAL_RAINFOREST"]
            elif latitude < SUBTROPICAL_LIMIT:
                if altitude > ALTITUDE_ALPINE_THRESHOLD:
                    climate_map[i, j] = climates["ALPINE"]
                elif altitude > ALTITUDE_TUNDRA_THRESHOLD:
                    climate_map[i, j] = climates["TUNDRA"]
                else:
                    climate_map[i, j] = climates["DESERT"]
            elif latitude < TEMPERATE_LIMIT:
                if altitude > ALTITUDE_ALPINE_THRESHOLD:
                    climate_map[i, j] = climates["ALPINE"]
                elif altitude > ALTITUDE_TUNDRA_THRESHOLD:
                    climate_map[i, j] = climates["TUNDRA"]
                elif latitude > 45 and altitude < 500:
                    climate_map[i, j] = climates["MEDITERRANEAN"]
                else:
                    climate_map[i, j] = climates["TEMPERATE_FOREST"]
            else:
                if altitude > ALTITUDE_ALPINE_THRESHOLD:
                    climate_map[i, j] = climates["ALPINE"]
                elif altitude > ALTITUDE_TUNDRA_THRESHOLD:
                    climate_map[i, j] = climates["TUNDRA"]
                else:
                    climate_map[i, j] = climates["POLAR"]

    # visualize_climate_map(climate_map)
    return climate_map


# Function to visualize the climate map
def visualize_climate_map(climate_map):
    # Dictionary for climate color mapping

    # Initialize an empty color map (3D array for RGB values)
    color_map = np.zeros((climate_map.shape[0], climate_map.shape[1], 3))

    # Loop over the climate map and assign colors
    for i in range(climate_map.shape[0]):
        for j in range(climate_map.shape[1]):
            climate = climate_map[i, j]
            color = climate_colors.get(
                climate, [0, 0, 0]
            )  # Default to black if undefined
            color_map[i, j] = color  # Assign the RGB color

    plt.imshow(color_map)
    plt.title("Climate Map")
    plt.axis("off")
    plt.show()


# Example heightmap (altitude in meters)
# heightmap = np.random.randint(0, 4000, (100, 100))  # Random heightmap
# latitudes = np.linspace(
#     0, 90, 100
# )  # Simulated latitudes from equator (0) to poles (90)
#
# # Generate the climate map
# climate_map = generate_climate_map(heightmap, latitudes)
#
# # Visualize the climate map
# visualize_climate_map(climate_map)
