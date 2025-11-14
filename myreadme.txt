# Installer le code sur un autre PC
# 1. Cloner le projet
git clone https://github.com/Aktawind/super-enhancer.git
cd super-enhancer

# 2. Créer un environnement virtuel
python -m venv venv
.\venv\Scripts\activate

# 3. Installer les mêmes dépendances
pip install --upgrade pip
pip install wheel setuptools
pip install -r requirements.txt

# 4. Lancer l'application
python main.py

# Pour tester la librairie realesrgan unitairement
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --outscale 4 --fp32

# Pour générer l'exécutable de l'application
pyinstaller --noconfirm --windowed --name SuperEnhancerPro --icon="assets/app_icon.ico" --collect-all basicsr --collect-all gfpgan --collect-all realesrgan --add-data "assets;assets" --add-data "weights;weights" main.py