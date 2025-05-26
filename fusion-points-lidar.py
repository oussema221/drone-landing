import numpy as np
import cv2
from PIL import Image

 
 
image_path = r"C:\Users\USER\Documents\Desktop\pfa vers etoile\002.png"

#  convertir  image en RGB
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

# zones sûres (paved-area, grass, roof)
safe_colors = [
    (128, 64, 128),  # paved-area
    (0, 102, 0),     # grass
    (70, 70, 70)     # roof
]


mask_safe = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=bool)
for color in safe_colors:
    mask_safe |= np.all(image_np == color, axis=-1) #masque boolen pour zones sures


M = mask_safe.astype(np.uint8) #convertie en mask binaire
M = cv2.resize(M, (256, 256), interpolation=cv2.INTER_NEAREST) #redimensionner

# Supprimer les zones avec moins de 50 pixels
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(M, connectivity=8)
M_filtered = np.zeros_like(M)
for label in range(1, num_labels):
    if stats[label, cv2.CC_STAT_AREA] >= 50:  # Garder les régions avec 50 pixels ou plus
        M_filtered[labels == label] = 1
M = M_filtered  # Remplacer M par le masque filtré



# Calibration et projection des points LiDAR 
K = np.array([[3636.36, 0, 2736],
              [0, 3636.36, 1824],
              [0, 0, 1]])
R = np.array([[1, 0, 0],
              [0, 0.999, -0.052],
              [0, 0.052, 0.999]])
T = np.array([[0.15], [0], [0.05]])

# Trois points LiDAR exemple
lidar_points = [
    [-0.848, -0.438, 0.974],  # Point 1
    [-0.846, -0.436, 0.990],  # Point 2
    [-0.850, -0.440, 0.960]   # Point 3
]

#  projeter un point LiDAR sur l'image
def project_lidar_to_image(lidar_point):
    point_3d = np.array([[lidar_point[0]], [lidar_point[1]], [lidar_point[2]]])
    camera_position = R @ point_3d + T
    image_position = K @ camera_position
    u = image_position[0, 0] / image_position[2, 0]
    v = image_position[1, 0] / image_position[2, 0]
    return int(u), int(v)

# Fusionner les points LiDAR avec le masque
def fuse_lidar_with_mask(lidar_points, mask):
    lidar_on_mask = []
    for i, point in enumerate(lidar_points):
        u, v = project_lidar_to_image(point)
        if 0 <= u < 256 and 0 <= v < 256:
            if mask[v, u] == 1:
                lidar_on_mask.append((u, v, point[2]))
                print(f"Point LiDAR {i+1} ({point[0]}, {point[1]}, {point[2]}) -> (u={u}, v={v}) : M[v,u]=1")
            else:
                print(f"Point LiDAR {i+1} ({point[0]}, {point[1]}, {point[2]}) -> (u={u}, v={v}) : M[v,u]=0")
        else:
            print(f"Point LiDAR {i+1} ({point[0]}, {point[1]}, {point[2]}) -> (u={u}, v={v}) : Hors de l'image")
    return lidar_on_mask

# Exécuter la fusion
lidar_on_mask = fuse_lidar_with_mask(lidar_points, M)

# separation des region
num_labels, labels = cv2.connectedComponents(M, connectivity=4)
regions = []
for label in range(1, num_labels):
    region_mask = (labels == label).astype(np.uint8)
    regions.append(region_mask)
print(f"Nombre de régions trouvées : {len(regions)}")

# Calculer le score 
def calculate_score(region, lidar_points_on_region):

    if len(lidar_points_on_region) > 1:
        z_values = [p[2] for p in lidar_points_on_region]
        flatness = 1 / (1 + np.std(z_values))  
    else:
        flatness = 0.1  # Valeur si nombre de ponts inferieur a 1

    # Taille 
    size = np.sum(region)
    size_score = size / (256 * 256)

    # Distance au centre
    moments = cv2.moments(region)
    if moments["m00"] != 0:  # Vérification de region n est pas vide 
        cx = int(moments["m10"] / moments["m00"])  # Coordonnée x du centre
        cy = int(moments["m01"] / moments["m00"])  # Coordonnée y du centre
        distance = np.sqrt((cx - 128)**2 + (cy - 128)**2)  # Distance au centre
        max_distance = np.sqrt(128**2 + 128**2)  # Distance maximale
        distance_score = 1 - (distance / max_distance)  # Normalisé entre 0 et 1
    else:
        distance_score = 0.1  

    # Score final 
    final_score = 0.4 * flatness + 0.3 * size_score + 0.3 * distance_score
    return final_score

# Calculer les scores
best_region_idx = -1
best_score = -1
for idx, region in enumerate(regions):
    # Trouver les points LiDAR dans la région
    points_in_region = []
    for u, v, z in lidar_on_mask:
        if region[v, u] == 1:
            points_in_region.append((u, v, z))
    
    # Calculer le score
    score = calculate_score(region, points_in_region)
    print(f"Région {idx + 1} : Score = {score:.3f} (Points LiDAR : {len(points_in_region)})")
    
    # Mettre à jour la meilleure région
    if score > best_score:
        best_score = score
        best_region_idx = idx

# Afficher la meilleure région
if best_region_idx != -1:
    print(f"Meilleure région : Région {best_region_idx + 1} avec un score de {best_score:.3f}")
else:
    print("Aucune région trouvée.") 