from typing import Dict, List, Tuple
import ollama
from realm_area import RealmArea


class RoomDescriptionGenerator:
    def __init__(self, realm_data: Dict, model_name: str = "qwen2.5:14b"):
        self.realm_data = realm_data
        self.model_name = model_name
        self.biome_names = {
            0: "Deep Water",
            1: "Shallow Water",
            2: "Beach",
            3: "Plains",
            4: "Forest",
            5: "Hills",
            6: "Mountains",
        }
        self.areas = self.realm_data["areas"]

    def define_areas(self) -> Dict[Tuple[int, int, int, int], RealmArea]:
        """Define different areas of the realm."""
        areas = {}

        # Example area definitions
        areas[(0, 0, 50, 50)] = RealmArea(
            "Misty Highlands",
            "A region of rolling hills shrouded in mist, known for its mysterious ruins and hidden valleys.",
            {"atmosphere": "mysterious", "vegetation": "sparse", "wildlife": "elusive"},
        )
        areas[(50, 0, 100, 50)] = RealmArea(
            "Sunlit Meadows",
            "Vast plains of swaying grass and wildflowers, dotted with small ponds and groves.",
            {"atmosphere": "peaceful", "vegetation": "lush", "wildlife": "abundant"},
        )
        # Add more areas as needed

        return areas

    def get_surrounding_info(self, x: int, y: int, radius: int = 2) -> Dict:
        """Gather information about the surrounding area."""
        surroundings = {"biomes": [], "rivers": False, "villages": [], "poi": []}

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = (x + dx) % self.realm_data["biome_map"].shape[0], (
                    y + dy
                ) % self.realm_data["biome_map"].shape[1]

                surroundings["biomes"].append(
                    self.biome_names[self.realm_data["biome_map"][nx][ny]]
                )

                if self.realm_data["rivers"][nx][ny]:
                    surroundings["rivers"] = True

                for vx, vy in self.realm_data["villages"]:
                    if (nx, ny) == (vx, vy):
                        surroundings["villages"].append((dx, dy))

                for px, py, poi_type in self.realm_data["points_of_interest"]:
                    if (nx, ny) == (px, py):
                        surroundings["poi"].append((dx, dy, poi_type))

        return surroundings

    def get_area(self, x: int, y: int) -> RealmArea | None:
        """Determine which area the given coordinates belong to."""
        for (x1, y1, x2, y2), area in self.areas.items():
            if x1 <= x < x2 and y1 <= y < y2:
                return area
        return None  # Default area or handle this case as needed

    def generate_room_description(self, x: int, y: int) -> str:
        """Generate a layered room description incorporating area characteristics."""
        biome = self.biome_names[self.realm_data["biome_map"][x][y]]
        height = self.realm_data["heightmap"][x][y]
        is_river = self.realm_data["rivers"][x][y]
        surroundings = self.get_surrounding_info(x, y)
        area = self.get_area(x, y)

        # Prepare the prompt for Ollama
        prompt = f"""
        Generate a detailed, immersive room description for a text-based multiplayer game (MUD). The description should have three parts:
        1. A vivid description of the immediate area (the room itself).
        2. A broader description of the surrounding region and any notable landmarks or features.
        3. How this room fits into the larger area of the realm.

        Use the following information:
        - Current biome: {biome}
        - Elevation: {height:.2f} (0 is sea level, higher is more elevated)
        - River present: {'Yes' if is_river else 'No'}
        - Nearby biomes: {', '.join(set(surroundings['biomes']))}
        - Nearby features:
          * Villages: {len(surroundings['villages'])}
          * Points of Interest: {[poi[2] for poi in surroundings['poi']]}
        - Area: {area.name if area else 'Unknown'}
        - Area description: {area.description if area else 'N/A'}
        - Area characteristics: {area.characteristics if area else 'N/A'}

        The description should be engaging and hint at potential adventures or points of interest. Use sensory details and evocative language to bring the scene to life. Incorporate the area's unique characteristics into the description.
        """

        # * Atmosphere: {area.characteristics.get('atmosphere', 'N/A') if area else 'N/A'}
        # * Vegetation: {area.characteristics.get('vegetation', 'N/A') if area else 'N/A'}
        # * Wildlife: {area.characteristics.get('wildlife', 'N/A') if area else 'N/A'}

        try:
            # Call Ollama to generate the description
            response = ollama.generate(model=self.model_name, prompt=prompt)
            return response["response"]
        except Exception as e:
            return f"Error generating description: {str(e)}"
