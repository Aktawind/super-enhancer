import customtkinter as ctk
from PIL import Image
import cv2
import os
import torch
from tkinter import filedialog, messagebox

# Importations de Real-ESRGAN
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
### GFPGAN ### Importation de la librairie de restauration de visages
from gfpgan import GFPGANer

PREVIEW_SIZE = (400, 400)

class UpscalerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Super Enhancer Pro")
        self.geometry("1000x600")

        try:
            self.iconbitmap('assets/app_icon.ico')
        except Exception as e:
            print(f"Avertissement : Impossible de charger l'icône - {e}")

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.input_path = None
        self.output_pil_image = None
        self.realesrgan_model = None
        ### GFPGAN ### Variable pour stocker le modèle de visage
        self.gfpgan_model = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.controls_frame = ctk.CTkFrame(self, width=180, corner_radius=0)
        self.controls_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.controls_frame.grid_rowconfigure(6, weight=1) # On ajoute une ligne pour la checkbox

        try:
            logo_pil = Image.open("assets/app_logo.png")
            logo_image = ctk.CTkImage(light_image=logo_pil, size=(140, int(140 * logo_pil.height / logo_pil.width)))
            self.logo_image_label = ctk.CTkLabel(self.controls_frame, image=logo_image, text="")
            self.logo_image_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        except Exception as e:
            self.logo_image_label = ctk.CTkLabel(self.controls_frame, text="Super Enhancer", font=ctk.CTkFont(size=20, weight="bold"))
            self.logo_image_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.controls_title = ctk.CTkLabel(self.controls_frame, text="Contrôles", font=ctk.CTkFont(size=16))
        self.controls_title.grid(row=1, column=0, padx=20, pady=(10, 0))

        self.btn_load = ctk.CTkButton(self.controls_frame, text="1. Charger une image", command=self.load_image)
        self.btn_load.grid(row=2, column=0, padx=20, pady=10)

        self.btn_process = ctk.CTkButton(self.controls_frame, text="2. Améliorer l'image", command=self.process_image, state="disabled")
        self.btn_process.grid(row=3, column=0, padx=20, pady=10)

        ### GFPGAN ### La checkbox pour activer la restauration des visages
        self.cb_face_restore = ctk.CTkCheckBox(self.controls_frame, text="Restaurer les visages")
        self.cb_face_restore.grid(row=4, column=0, padx=20, pady=10)
        self.cb_face_restore.select() # Cochée par défaut

        self.btn_save = ctk.CTkButton(self.controls_frame, text="3. Enregistrer", command=self.save_image, state="disabled")
        self.btn_save.grid(row=5, column=0, padx=20, pady=10)

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

        self.separator = ctk.CTkFrame(self.images_frame, width=2, fg_color="gray")
        self.separator.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="ns")

        self.status_label = ctk.CTkLabel(self, text="Prêt. En attente du chargement des modèles...", height=25)
        self.status_label.grid(row=1, column=1, padx=20, pady=(0,10), sticky="ew")

        self.after(100, self.load_models)

    def load_models(self):
        """Charge les deux modèles : RealESRGAN et GFPGAN."""
        self.update_status("Chargement des modèles... Veuillez patienter.")
        try:
            # Chargement de RealESRGAN
            model_path_realesrgan = os.path.join('weights', 'RealESRGAN_x4plus.pth')
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.realesrgan_model = RealESRGANer(scale=4, model_path=model_path_realesrgan, model=model, tile=0, tile_pad=10, pre_pad=0, half=torch.cuda.is_available())

            # ### GFPGAN ### Chargement de GFPGAN
            model_path_gfpgan = os.path.join('weights', 'GFPGANv1.4.pth')
            self.gfpgan_model = GFPGANer(model_path=model_path_gfpgan, upscale=4, arch='clean', channel_multiplier=2, bg_upsampler=self.realesrgan_model)

            self.update_status("Modèles chargés. Prêt à améliorer des images.")
        except Exception as e:
            messagebox.showerror("Erreur de chargement de modèle", f"Assurez-vous que les fichiers 'RealESRGAN_x4plus.pth' et 'GFPGANv1.4.pth' sont dans le dossier 'weights'.\n\nErreur: {e}")
            self.quit()

    def process_image(self):
        """Lance le processus d'amélioration, avec ou sans GFPGAN."""
        if not self.input_path or not self.realesrgan_model:
            return

        self.update_status("Amélioration en cours... Cette opération peut être longue.")
        self.btn_process.configure(state="disabled")
        self.update()

        try:
            # --- LA CORRECTION EST ICI ---
            # On charge l'image en forçant le mode couleur (3 canaux) pour être compatible avec les modèles IA
            img = cv2.imread(self.input_path, cv2.IMREAD_COLOR)

            if self.cb_face_restore.get() and self.gfpgan_model:
                self.update_status("Étape 1/2 : Amélioration générale...")
                _, _, restored_img = self.gfpgan_model.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                output = restored_img
            else:
                self.update_status("Amélioration simple (sans restauration des visages)...")
                output, _ = self.realesrgan_model.enhance(img, outscale=4)

            self.output_pil_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

            ctk_output_image = ctk.CTkImage(light_image=self.output_pil_image, size=(PREVIEW_SIZE[0], int(PREVIEW_SIZE[1] * self.output_pil_image.size[1] / self.output_pil_image.size[0])))
            self.output_image_label.configure(image=ctk_output_image, text="")

            self.btn_save.configure(state="normal")
            self.update_status("Amélioration terminée ! L'image est prête à être enregistrée.")

        except Exception as e:
            self.update_status("Erreur lors du traitement.")
            messagebox.showerror("Erreur de traitement", str(e))
        finally:
            self.btn_process.configure(state="normal")

    # ... Les autres fonctions (load_image, save_image, etc.) ne changent pas ...
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path: return
        self.input_path = path
        self.output_pil_image = None
        self.output_image_label.configure(image=None, text="Le résultat apparaîtra ici.")
        self.btn_save.configure(state="disabled")
        try:
            pil_image = Image.open(path)
            ctk_image = ctk.CTkImage(light_image=pil_image, size=(PREVIEW_SIZE[0], int(PREVIEW_SIZE[1] * pil_image.size[1] / pil_image.size[0])))
            self.input_image_label.configure(image=ctk_image, text="")
            self.btn_process.configure(state="normal")
            self.update_status(f"Image chargée : {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Erreur d'ouverture", f"Impossible de charger l'image.\n{e}")

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