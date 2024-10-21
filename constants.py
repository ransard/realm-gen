biomes = {
    "DEEP_WATER": 0,
    "SHALLOW_WATER": 1,
    "BEACH": 2,
    "PLAINS": 3,
    "FOREST": 4,
    "HILLS": 5,
    "MOUNTAINS": 6,
}
biome_names = {v: k for k, v in biomes.items()}
biome_colors = {
    0: (0, 0, 0.5),  # Deep Water: Dark Blue
    1: (0, 0, 1),  # Shallow Water: Blue
    2: (1, 1, 0.7),  # Beach: Pale Yellow
    3: (0.5, 0.8, 0.5),  # Plains: Light Green
    4: (0, 0.5, 0),  # Forest: Dark Green
    5: (0.5, 0.5, 0.5),  # Hills: Gray
    6: (0.7, 0.7, 0.7),  # Mountains: Light Gray
}
model_name = "llama3.1"
