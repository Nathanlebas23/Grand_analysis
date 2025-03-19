import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Définition des centres et des rayons
centers = [(10, 10), (30, 20), (20, 40)]
rayons = [2, 4, 8]

# Calcul des valeurs minimales et maximales des rayons pour la normalisation
min_r = min(rayons)
max_r = max(rayons)

fig, ax = plt.subplots()

# Pour chaque centre et rayon, on calcule la couleur à partir du colormap "Oranges"
for center, r in zip(centers, rayons):
    # Normalisation : pour le rayon minimum (2) -> 0 (orange pâle)
    # et pour le rayon maximum (8) -> 1 (orange complet)
    normalized = (r - min_r) / (max_r - min_r)
    color = plt.cm.Oranges(normalized)
    circle = Circle(center, r, facecolor=color, edgecolor='black')
    ax.add_patch(circle)

# Ajustement des limites de l'axe pour afficher tous les cercles
ax.set_xlim(0, 50)
ax.set_ylim(0, 50)
ax.set_aspect('equal', adjustable='box')

plt.title("Cercles remplis avec gradient de couleur 'Oranges'")
plt.show()
