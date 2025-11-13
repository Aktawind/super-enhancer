# 1. Cloner le projet
git clone <url_de_votre_projet>
cd super-enhancer

# 2. Créer un environnement virtuel
python -m venv venv
.\venv\Scripts\activate

# 3. Installer EXACTEMENT les mêmes dépendances
pip install -r requirements.txt

# 4. Lancer l'application !
python app.py

En ligne de commande :
Pour lancer la librairie :
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --outscale 4 --fp32

Pour générer l'exécutable :
pyinstaller --onefile --windowed --add-data "weights;weights" app.py