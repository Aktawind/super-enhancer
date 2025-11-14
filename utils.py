import os
import json
import cv2
import numpy as np
from PIL import Image

def resource_path(relative_path):
    """ Renvoie le chemin absolu vers une ressource, fonctionne pour le dev et pour PyInstaller """
    try:
        # PyInstaller crée un dossier temporaire et stocke son chemin dans _MEIPASS.
        base_path = sys._MEIPASS
    except Exception:
        # Si on n'est pas dans un exécutable, le chemin de base est le dossier de travail actuel
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def get_config_path():
    """Retourne le chemin vers le fichier de config dans le dossier AppData."""
    app_data_path = os.getenv('APPDATA')
    if not app_data_path:
        return "config.json"

    config_dir = os.path.join(app_data_path, "SuperEnhancerPro")
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "config.json")

def correct_colors_grey_world(pil_image):
    """Applique l'algorithme Grey World pour corriger la balance des blancs."""
    if pil_image is None:
        return None

    # On convertit l'image PIL en format OpenCV (NumPy array)
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Calcul des moyennes et des facteurs de mise à l'échelle
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3

    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r

    # Application de la correction
    img_corrected = img.copy()
    img_corrected[:, :, 0] = np.clip(img[:, :, 0] * scale_b, 0, 255)
    img_corrected[:, :, 1] = np.clip(img[:, :, 1] * scale_g, 0, 255)
    img_corrected[:, :, 2] = np.clip(img[:, :, 2] * scale_r, 0, 255)

    # On reconvertit en PIL pour l'affichage
    return Image.fromarray(cv2.cvtColor(img_corrected, cv2.COLOR_BGR2RGB))