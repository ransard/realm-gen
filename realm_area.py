from typing import Dict, List, Tuple


class RealmArea:
    def __init__(self, name: str, description: str, characteristics: Dict[str, str]):
        self.name = name
        self.description = description
        self.characteristics = characteristics
