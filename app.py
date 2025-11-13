import customtkinter as ctk
from PIL import Image
import cv2
import os
import torch
from tkinter import filedialog, messagebox
import threading
import queue
import time
import json

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

PREVIEW_SIZE = (400, 400)

### SOLUTION EXE ### Nouvelle fonction pour trouver le dossier de configuration
def get_config_path():
    """Retourne le chemin vers le fichier de config dans le dossier AppData."""
    app_data_path = os.getenv('APPDATA')
    if not app_data_path: # Si la variable n'existe pas, on se rabat sur le dossier local
        return "config.json"

    # On crée un dossier pour notre application dans AppData si il n'existe pas
    config_dir = os.path.join(app_data_path, "SuperEnhancerPro")
    os.makedirs(config_dir, exist_ok=True)

    return os.path.join(config_dir, "config.json")

class UpscalerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # ... (le début est identique)
        self.title("Super Enhancer Pro")
        self.geometry("1000x650")
        try:
            self.iconbitmap('assets/app_icon.ico')
        except Exception:
            pass
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        self.input_path = None
        self.output_pil_image = None
        self.realesrgan_model = None
        self.gfpgan_model = None
        self.progress_queue = queue.Queue()
        self.seconds_per_megapixel = 0
        self.processing_start_time = 0
        self.estimated_total_time = 0

        ### SOLUTION BUG IMAGE ### Attributs pour garder les références aux images
        self.ctk_input_image = None
        self.ctk_output_image = None

        ### SOLUTION EXE ### On utilise la nouvelle fonction pour le chemin
        self.config_file = get_config_path()
        self.load_config()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ... (le reste de l'interface dans __init__ est identique)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.controls_frame = ctk.CTkFrame(self, width=180, corner_radius=0)
        self.controls_frame.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.controls_frame.grid_rowconfigure(5, weight=1)
        try:
            logo_pil = Image.open("assets/app_logo.png")
            logo_image = ctk.CTkImage(light_image=logo_pil, size=(140, int(140 * logo_pil.height / logo_pil.width)))
            ctk.CTkLabel(self.controls_frame, image=logo_image, text="").grid(row=0, column=0, padx=20, pady=(20, 10))
        except Exception:
            ctk.CTkLabel(self.controls_frame, text="Super Enhancer", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10))
        ctk.CTkLabel(self.controls_frame, text="Contrôles", font=ctk.CTkFont(size=16)).grid(row=1, column=0, padx=20, pady=(10, 0))
        self.btn_load = ctk.CTkButton(self.controls_frame, text="1. Charger une image", command=self.load_image)
        self.btn_load.grid(row=2, column=0, padx=20, pady=10)
        self.btn_process = ctk.CTkButton(self.controls_frame, text="2. Améliorer l'image", command=self.start_processing_thread, state="disabled")
        self.btn_process.grid(row=3, column=0, padx=20, pady=10)
        self.cb_face_restore = ctk.CTkCheckBox(self.controls_frame, text="Restaurer les visages")
        self.cb_face_restore.grid(row=4, column=0, padx=20, pady=10)
        self.cb_face_restore.select()
        self.btn_save = ctk.CTkButton(self.controls_frame, text="3. Enregistrer", command=self.save_image, state="disabled")
        self.btn_save.grid(row=5, column=0, padx=20, pady=10, sticky="s")
        self.images_frame = ctk.CTkFrame(self, corner_radius=10)
        self.images_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.images_frame.grid_columnconfigure((0, 2), weight=1, uniform="panels")
        self.images_frame.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(self.images_frame, text="Image Originale", font=ctk.CTkFont(size=16)).grid(row=0, column=0, pady=10)
        ctk.CTkLabel(self.images_frame, text="Image Améliorée", font=ctk.CTkFont(size=16)).grid(row=0, column=2, pady=10)
        self.input_image_label = ctk.CTkLabel(self.images_frame, text="Chargez une image...", corner_radius=10, fg_color=("gray90", "gray13"))
        self.input_image_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.output_image_label = ctk.CTkLabel(self.images_frame, text="Le résultat apparaîtra ici.", corner_radius=10, fg_color=("gray90", "gray13"))
        self.output_image_label.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")
        ctk.CTkFrame(self.images_frame, width=2, fg_color="gray").grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="ns")
        self.progress_frame = ctk.CTkFrame(self, corner_radius=10)
        self.progress_frame.grid(row=1, column=1, padx=20, pady=(0, 10), sticky="nsew")
        self.progress_frame.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(self.progress_frame, text="Prêt. En attente du chargement des modèles...")
        self.status_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(5,0), sticky="w")
        self.progressbar = ctk.CTkProgressBar(self.progress_frame)
        self.progressbar.grid(row=1, column=0, padx=10, pady=(0,5), sticky="ew")
        self.progressbar.set(0)
        self.time_label = ctk.CTkLabel(self.progress_frame, text="")
        self.time_label.grid(row=1, column=1, padx=10, pady=(0,5), sticky="e")

        self.after(100, self.load_models)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path: return
        self.input_path = path
        self.output_pil_image = None
        self.output_image_label.configure(image=None, text="Le résultat apparaîtra ici.")
        self.btn_save.configure(state="disabled")
        self.progressbar.set(0)
        self.time_label.configure(text="")
        try:
            pil_image = Image.open(path)
            # On stocke l'image dans self.ctk_input_image pour la protéger du garbage collector
            self.ctk_input_image = ctk.CTkImage(light_image=pil_image, size=(PREVIEW_SIZE[0], int(PREVIEW_SIZE[1] * pil_image.height / pil_image.width)))
            self.input_image_label.configure(image=self.ctk_input_image, text="")
            self.btn_process.configure(state="normal")
            self.update_status(f"Image chargée : {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Erreur d'ouverture", f"Impossible de charger l'image.\n{e}")

    def update_progress(self):
        elapsed_time = time.time() - self.processing_start_time

        if self.estimated_total_time > 0:
            progress = min(elapsed_time / self.estimated_total_time, 0.99)
            self.progressbar.set(progress)
            self.time_label.configure(text=f"{int(elapsed_time)}s / {int(self.estimated_total_time)}s")
        else:
            self.time_label.configure(text=f"Temps écoulé : {int(elapsed_time)}s")
            self.progressbar.set(elapsed_time / 60)

        try:
            message = self.progress_queue.get_nowait()
            msg_type, msg_data = message

            if msg_type == 'timing':
                duration, pixels = msg_data
                megapixels = pixels / 1_000_000
                if megapixels > 0:
                    self.seconds_per_megapixel = duration / megapixels
                    self.save_config()
            elif msg_type == 'done':
                self.output_pil_image = msg_data
                # On stocke l'image dans self.ctk_output_image pour la protéger
                self.ctk_output_image = ctk.CTkImage(light_image=self.output_pil_image, size=(PREVIEW_SIZE[0], int(PREVIEW_SIZE[1] * self.output_pil_image.height / self.output_pil_image.width)))
                self.output_image_label.configure(image=self.ctk_output_image, text="")

                self.update_status("Amélioration terminée !")
                self.progressbar.set(1)
                self.time_label.configure(text=f"Terminé en {int(elapsed_time)}s")

                self.btn_load.configure(state="normal")
                self.btn_process.configure(state="normal")
                self.btn_save.configure(state="normal")
                return
            elif msg_type == 'error':
                messagebox.showerror("Erreur de traitement", msg_data)
                self.update_status("Erreur lors du traitement.")
                self.progressbar.set(0)
                self.time_label.configure(text="")

                self.btn_load.configure(state="normal")
                self.btn_process.configure(state="normal")
                return

        except queue.Empty:
            pass

        self.after(100, self.update_progress)

    # ... (les autres fonctions sont identiques, copiez-les ici)
    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.seconds_per_megapixel = config.get("seconds_per_megapixel", 0)
                if self.seconds_per_megapixel > 0:
                    print(f"Vitesse de traitement chargée depuis la config : {self.seconds_per_megapixel:.2f} s/Mpx")
        except (FileNotFoundError, json.JSONDecodeError):
            self.seconds_per_megapixel = 0
            print("Aucun fichier de config trouvé. La vitesse sera calculée au premier traitement.")

    def save_config(self):
        config = {"seconds_per_megapixel": self.seconds_per_megapixel}
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)
        print("Configuration sauvegardée.")

    def on_closing(self):
        if self.seconds_per_megapixel > 0:
            self.save_config()
        self.destroy()

    def start_processing_thread(self):
        self.btn_process.configure(state="disabled")
        self.btn_load.configure(state="disabled")
        self.btn_save.configure(state="disabled")
        try:
            with Image.open(self.input_path) as img:
                pixels = img.width * img.height
                megapixels = pixels / 1_000_000
        except Exception:
            megapixels = 1
        if self.seconds_per_megapixel > 0:
            self.estimated_total_time = megapixels * self.seconds_per_megapixel
            self.update_status(f"Amélioration en cours... Temps estimé : {int(self.estimated_total_time)}s")
        else:
            self.estimated_total_time = 0
            self.update_status("Amélioration en cours... (calcul du temps pour la première fois)")
        self.processing_start_time = time.time()
        thread = threading.Thread(target=self.process_image_threaded, args=(pixels,))
        thread.daemon = True
        thread.start()
        self.after(100, self.update_progress)

    def process_image_threaded(self, pixels):
        start_time = time.time()
        try:
            img = cv2.imread(self.input_path, cv2.IMREAD_COLOR)
            if self.cb_face_restore.get() and self.gfpgan_model:
                _, _, restored_img = self.gfpgan_model.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                output = restored_img
            else:
                output, _ = self.realesrgan_model.enhance(img, outscale=4)
            output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            duration = time.time() - start_time
            self.progress_queue.put(('timing', (duration, pixels)))
            self.progress_queue.put(('done', output_pil))
        except Exception as e:
            self.progress_queue.put(('error', str(e)))

    def load_models(self):
        self.update_status("Chargement des modèles... Veuillez patienter.")
        try:
            model_path_realesrgan = os.path.join('weights', 'RealESRGAN_x4plus.pth')
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.realesrgan_model = RealESRGANer(scale=4, model_path=model_path_realesrgan, model=model, tile=0, tile_pad=10, pre_pad=0, half=torch.cuda.is_available())
            model_path_gfpgan = os.path.join('weights', 'GFPGANv1.4.pth')
            self.gfpgan_model = GFPGANer(model_path=model_path_gfpgan, upscale=4, arch='clean', channel_multiplier=2, bg_upsampler=self.realesrgan_model)
            self.update_status("Modèles chargés. Prêt à améliorer des images.")
        except Exception as e:
            messagebox.showerror("Erreur de chargement de modèle", f"Assurez-vous que les fichiers 'RealESRGAN_x4plus.pth' et 'GFPGANv1.4.pth' sont dans le dossier 'weights'.\n\nErreur: {e}")
            self.quit()

    def save_image(self):
        if not self.output_pil_image: return
        input_dir, input_filename = os.path.split(self.input_path)
        name, ext = os.path.splitext(input_filename)
        default_name = f"{name}_enhanced.png"
        path = filedialog.asksaveasfilename(initialfile=default_name, defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            self.output_pil_image.save(path)
            self.update_status(f"Image enregistrée sous : {os.path.basename(path)}")
            messagebox.showinfo("Succès", "L'image a été enregistrée avec succès.")

    def update_status(self, text):
        self.status_label.configure(text=text)
        self.update_idletasks()

if __name__ == "__main__":
    app = UpscalerApp()
    app.mainloop()