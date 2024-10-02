from realm_generator import RealmGenerator
from room_generator import RoomDescriptionGenerator

# Usage
generator = RealmGenerator(width=512, height=512)
realm = generator.generate_realm()

# Print some information about the generated realm
print(f"Generated a realm of size {generator.width}x{generator.height}")
print(f"Number of villages: {len(realm['villages'])}")
print(f"Number of points of interest: {len(realm['points_of_interest'])}")

# You would typically save this realm data or use it to initialize your game world

print("Generated Realm Data:")
description_generator = RoomDescriptionGenerator(realm)

# Generate a description for a specific room
x, y = 50, 50  # Example coordinates
# room_name = description_generator.get_room_name(x, y)
room_description = description_generator.generate_room_description(x, y)

# print(f"Room Name: {room_name}")
print("Room Description:")
print(room_description)
