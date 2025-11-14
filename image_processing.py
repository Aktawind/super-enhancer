import cv2
import torch
from PIL import Image
import os

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

class ImageProcessor:
    def __init__(self):
        self.realesrgan_model = None
        self.gfpgan_model = None

    def load_models(self):
        """Charge les modèles RealESRGAN et GFPGAN."""
        print("Chargement des modèles...")
        try:
            model_path_realesrgan = os.path.join('weights', 'RealESRGAN_x4plus.pth')
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            bg_upsampler = RealESRGANer(scale=4, model_path=model_path_realesrgan, model=model, tile=400, tile_pad=10, pre_pad=0, half=torch.cuda.is_available())

            model_path_gfpgan = os.path.join('weights', 'GFPGANv1.4.pth')
            self.gfpgan_model = GFPGANer(model_path=model_path_gfpgan, upscale=4, arch='clean', channel_multiplier=2, bg_upsampler=bg_upsampler)

            # On stocke le modèle Real-ESRGAN de base aussi, pour le cas où GFPGAN n'est pas utilisé
            self.realesrgan_model = bg_upsampler
            print("Modèles chargés avec succès.")
            return True
        except Exception as e:
            print(f"Erreur de chargement de modèle : {e}")
            return e # On retourne l'erreur pour l'afficher dans l'UI

    def process_image(self, input_path, restore_faces, cancellation_event):
        """Améliore une image en utilisant les modèles chargés."""
        print(f"Traitement de {input_path}, restauration des visages: {restore_faces}")
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Impossible de lire le fichier image.")

        # On vérifie le signal AVANT de commencer le gros travail
        if cancellation_event.is_set():
            print("Traitement annulé avant le démarrage.")
            return "cancelled"

        if restore_faces and self.gfpgan_model:
            _, _, restored_img = self.gfpgan_model.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            output = restored_img
        elif self.realesrgan_model:
            output, _ = self.realesrgan_model.enhance(img, outscale=4)
        else:
            raise RuntimeError("Aucun modèle n'est chargé pour le traitement.")

        # On vérifie une dernière fois après la fin du traitement
        if cancellation_event.is_set():
            print("Traitement annulé après la fin.")
            return "cancelled"

        return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))