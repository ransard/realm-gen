# biomes = {
#     "DEEP_WATER": 0,
#     "SHALLOW_WATER": 1,
#     "BEACH": 2,
#     "PLAINS": 3,
#     "FOREST": 4,
#     "HILLS": 5,
#     "MOUNTAINS": 6,
# }
# biome_names = {v: k for k, v in biomes.items()}
# biome_colors = {
#     0: (0, 0, 0.5),  # Deep Water: Dark Blue
#     1: (0, 0, 1),  # Shallow Water: Blue
#     2: (1, 1, 0.7),  # Beach: Pale Yellow
#     3: (0.5, 0.8, 0.5),  # Plains: Light Green
#     4: (0, 0.5, 0),  # Forest: Dark Green
#     5: (0.5, 0.5, 0.5),  # Hills: Gray
#     6: (0.7, 0.7, 0.7),  # Mountains: Light Gray
# }

climates = {
    "TROPICAL_RAINFOREST": 0,
    "SAVANNA": 1,
    "DESERT": 2,
    "MEDITERRANEAN": 3,
    "TEMPERATE_FOREST": 4,
    "CONTINENTAL": 5,
    "TUNDRA": 6,
    "ALPINE": 7,
    "POLAR": 8,
}

climate_names = {v: k for k, v in climates.items()}

climate_colors = {
    0: (0, 1, 0),  # Tropical Rainforest: Green
    1: (1, 0.65, 0),  # Savanna: Orange
    2: (1, 1, 0),  # Desert: Yellow
    3: (0.5, 1, 0.5),  # Mediterranean: Light green
    4: (0, 0.5, 0),  # Temperate Forest: Dark green
    5: (0.5, 0.75, 1),  # Continental: Light blue
    6: (1, 1, 1),  # Tundra: White
    7: (0.5, 0.5, 0.5),  # Alpine: Gray
    8: (0, 0, 1),  # Polar: Blue
}

model_name = "llama3.1"
