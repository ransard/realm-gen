import matplotlib.pyplot as plt
import os
from constants import climate_colors

output_folder = "output"


class ImageHandler:
    def __init__(self, biome_colors):
        self.biome_colors = biome_colors

        # check if output_folder exists otherwise create it
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def save_image(self, data, filename, cmap="viridis", custom_cmap=None):
        plt.figure(figsize=(10, 10))
        if custom_cmap:
            cmap = plt.matplotlib.colors.ListedColormap(list(custom_cmap.values()))
        plt.imshow(data, cmap=cmap)
        plt.colorbar()
        plt.title(filename.split(".")[0].replace("_", " ").title())
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{filename}")
        plt.close()

    def save_image_with_points(self, background, points, filename, title):
        plt.figure(figsize=(10, 10))
        plt.imshow(
            background,
            cmap=plt.matplotlib.colors.ListedColormap(list(climate_colors.values())),
        )
        x, y = zip(*points)
        plt.scatter(y, x, c="red", s=20)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{filename}")
        plt.close()

    def save_image_with_areas(self, background, areas, filename):
        plt.figure(figsize=(10, 10))
        plt.imshow(
            background,
            cmap=plt.matplotlib.colors.ListedColormap(list(climate_colors.values())),
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
        plt.savefig(f"{output_folder}/{filename}")
        plt.close()
