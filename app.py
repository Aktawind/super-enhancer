import customtkinter as ctk
from PIL import Image
import cv2
import os
import torch
from tkinter import filedialog, messagebox

# Importations de Real-ESRGAN
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

PREVIEW_SIZE = (400, 400)

class UpscalerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Configuration de la fenêtre principale ---
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
        self.model = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Frame de gauche pour les contrôles (modifiée pour le logo) ---
        self.controls_frame = ctk.CTkFrame(self, width=180, corner_radius=0)
        self.controls_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.controls_frame.grid_rowconfigure(5, weight=1) # L'espace flexible est maintenant à la ligne 5

        # --- AMÉLIORATION 3 : Ajouter un logo ---
        try:
            # On charge l'image PNG avec PIL
            logo_pil = Image.open("assets/app_logo.png")

            # On la convertit en objet CTkImage, en la redimensionnant pour qu'elle s'adapte
            # à la largeur du panneau (140px pour laisser 20px de marge de chaque côté)
            logo_image = ctk.CTkImage(
                light_image=logo_pil,
                size=(140, int(140 * logo_pil.height / logo_pil.width)) # Calcule la hauteur pour garder le ratio
            )
            # On crée un label pour afficher l'image (sans texte)
            self.logo_image_label = ctk.CTkLabel(self.controls_frame, image=logo_image, text="")
            self.logo_image_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        except Exception as e:
            print(f"Avertissement: Impossible de charger le logo 'app_logo.png' - {e}")
            # Si le logo n'est pas trouvé, on affiche un titre texte à la place
            self.logo_image_label = ctk.CTkLabel(self.controls_frame, text="Super Enhancer", font=ctk.CTkFont(size=20, weight="bold"))
            self.logo_image_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # On décale les autres contrôles vers le bas
        self.controls_title = ctk.CTkLabel(self.controls_frame, text="Contrôles", font=ctk.CTkFont(size=16))
        self.controls_title.grid(row=1, column=0, padx=20, pady=(10, 0))

        self.btn_load = ctk.CTkButton(self.controls_frame, text="1. Charger une image", command=self.load_image)
        self.btn_load.grid(row=2, column=0, padx=20, pady=10)

        self.btn_process = ctk.CTkButton(self.controls_frame, text="2. Améliorer l'image", command=self.process_image, state="disabled")
        self.btn_process.grid(row=3, column=0, padx=20, pady=10)

        self.btn_save = ctk.CTkButton(self.controls_frame, text="3. Enregistrer", command=self.save_image, state="disabled")
        self.btn_save.grid(row=4, column=0, padx=20, pady=10)

        # --- Le reste de l'interface (panneau d'images, barre de statut) ne change pas ---
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

        self.status_label = ctk.CTkLabel(self, text="Prêt. En attente du chargement du modèle...", height=25)
        self.status_label.grid(row=1, column=1, padx=20, pady=(0,10), sticky="ew")

        self.after(100, self.load_model)

    # ... (le reste du code, de load_model à la fin, est identique à la version précédente) ...

    def load_model(self):
        self.update_status("Chargement du modèle... Veuillez patienter.")
        try:
            model_path = os.path.join('weights', 'RealESRGAN_x4plus.pth')
            if not os.path.exists(model_path):
                messagebox.showerror("Erreur", f"Modèle non trouvé : {model_path}")
                self.quit()
                return
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.model = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=torch.cuda.is_available())
            self.update_status("Modèle chargé. Prêt à améliorer des images.")
        except Exception as e:
            messagebox.showerror("Erreur de chargement du modèle", str(e))
            self.quit()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            return
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

    def process_image(self):
        if not self.input_path or not self.model:
            return
        self.update_status("Amélioration en cours... Cette opération peut être longue.")
        self.btn_process.configure(state="disabled")
        self.update()
        try:
            img = cv2.imread(self.input_path, cv2.IMREAD_UNCHANGED)
            output, _ = self.model.enhance(img, outscale=4)
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

    def save_image(self):
        if not self.output_pil_image:
            return
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