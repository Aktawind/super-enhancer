# colab_runner.py
import os
import argparse
from PIL import Image
from image_processing import ImageProcessor

def main():
    # 1. Configurer les arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Lance le traitement d'image Super Enhancer en ligne de commande.")
    parser.add_argument('--input', type=str, required=True, help="Chemin vers l'image d'entrée.")
    parser.add_argument('--output_folder', type=str, default='results', help="Dossier où sauvegarder le résultat.")
    parser.add_argument('--no_face_enhance', action='store_true', help="Désactive la restauration des visages (GFPGAN).")
    args = parser.parse_args()

    # 2. Créer le processeur d'image et charger les modèles
    processor = ImageProcessor()
    result = processor.load_models()
    if result is not True:
        print(f"Erreur fatale lors du chargement des modèles : {result}")
        return

    # 3. Lancer le traitement de l'image
    print(f"Début du traitement pour : {args.input}")
    restore_faces = not args.no_face_enhance

    try:
        # Création d'un faux "événement d'annulation" car notre processeur l'attend
        class DummyEvent:
            def is_set(self): return False

        result_image = processor.process_image(args.input, restore_faces, DummyEvent())

        if result_image == "cancelled":
             print("Le traitement a été annulé (ceci ne devrait pas arriver dans ce script).")
             return

    except Exception as e:
        print(f"Une erreur est survenue pendant le traitement : {e}")
        return

    # 4. Sauvegarder le résultat
    os.makedirs(args.output_folder, exist_ok=True)
    filename = os.path.basename(args.input)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(args.output_folder, f"{name}_enhanced.png")

    result_image.save(output_path)
    print(f"Traitement terminé ! L'image a été sauvegardée ici : {output_path}")

if __name__ == "__main__":
    main()