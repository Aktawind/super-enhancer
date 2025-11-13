# ui.py
import customtkinter as ctk
from PIL import Image
from tkinter import filedialog, messagebox
import threading
import queue
import time
import os
import json

from image_processing import ImageProcessor
import utils

PREVIEW_SIZE = (400, 400)

class UpscalerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.processor = ImageProcessor()
        self.title("Super Enhancer Pro")
        self.geometry("1000x700")
        try:
            self.iconbitmap('assets/app_icon.ico')
        except Exception: pass
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # --- Attributs d'état ---
        self.input_path = None
        self.output_pil_image = None
        self.displayed_output_image = None
        self.ctk_input_image = None
        self.ctk_output_image = None # Référence permanente pour l'image de sortie

        # --- Threading & Config ---
        self.progress_queue = queue.Queue()
        self.seconds_per_megapixel = 0
        self.processing_start_time = 0
        self.estimated_total_time = 0
        self.config_file = utils.get_config_path()

        self.load_config()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.create_widgets()
        self.after(100, self.load_models_threaded)

    def create_widgets(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Panneau de contrôle ---
        self.controls_frame = ctk.CTkFrame(self, width=180, corner_radius=0)
        self.controls_frame.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.controls_frame.grid_rowconfigure(9, weight=1)

        try:
            logo_pil = Image.open("assets/app_logo.png")
            logo_image = ctk.CTkImage(light_image=logo_pil, size=(140, int(140 * logo_pil.height / logo_pil.width)))
            ctk.CTkLabel(self.controls_frame, image=logo_image, text="").grid(row=0, column=0, padx=20, pady=(20, 10))
        except Exception:
            ctk.CTkLabel(self.controls_frame, text="Super Enhancer", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10))

        self.btn_load = ctk.CTkButton(self.controls_frame, text="1. Charger l'image", command=self.load_image)
        self.btn_load.grid(row=1, column=0, padx=20, pady=10)
        self.btn_process = ctk.CTkButton(self.controls_frame, text="2. Améliorer", command=self.start_processing_thread, state="disabled")
        self.btn_process.grid(row=2, column=0, padx=20, pady=10)
        self.cb_face_restore = ctk.CTkCheckBox(self.controls_frame, text="Restaurer les visages")
        self.cb_face_restore.grid(row=3, column=0, padx=20, pady=10)
        self.cb_face_restore.select()

        ctk.CTkLabel(self.controls_frame, text="Correction du Jaunissement", font=ctk.CTkFont(weight="bold")).grid(row=4, column=0, padx=20, pady=(15, 0))
        self.color_slider = ctk.CTkSlider(self.controls_frame, from_=0, to=100, command=self.update_slider_label)
        self.color_slider.grid(row=5, column=0, padx=20, pady=(5, 0), sticky="ew")
        self.color_slider.set(0)
        self.color_slider.configure(state="disabled")
        self.color_slider_label = ctk.CTkLabel(self.controls_frame, text="Intensité : 0%")
        self.color_slider_label.grid(row=6, column=0, padx=20, pady=(0, 0))
        self.btn_apply_color = ctk.CTkButton(self.controls_frame, text="Appliquer", command=self.apply_manual_color, state="disabled")
        self.btn_apply_color.grid(row=7, column=0, padx=20, pady=(5, 5))
        self.btn_auto_color = ctk.CTkButton(self.controls_frame, text="Correction Auto", command=self.apply_auto_color, state="disabled")
        self.btn_auto_color.grid(row=8, column=0, padx=20, pady=(0, 5))
        self.btn_save = ctk.CTkButton(self.controls_frame, text="3. Enregistrer", command=self.save_image, state="disabled")
        self.btn_save.grid(row=9, column=0, padx=20, pady=10, sticky="s")

        # --- Panneau d'images et de progression ---
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

    def load_models_threaded(self):
        self.update_status("Chargement des modèles... Veuillez patienter.")
        thread = threading.Thread(target=self._load_models_worker)
        thread.daemon = True
        thread.start()

    def _load_models_worker(self):
        result = self.processor.load_models()
        if result is True:
            self.update_status("Modèles chargés. Prêt à améliorer des images.")
        else:
            self.progress_queue.put(('error', f"Erreur de chargement de modèle: {result}"))

    def start_processing_thread(self):
        for btn in [self.btn_process, self.btn_load, self.btn_save, self.btn_auto_color, self.btn_apply_color]: btn.configure(state="disabled")
        self.color_slider.configure(state="disabled")
        try:
            with Image.open(self.input_path) as img: pixels = img.width * img.height; megapixels = pixels / 1_000_000
        except Exception: megapixels = 1
        if self.seconds_per_megapixel > 0: self.estimated_total_time = megapixels * self.seconds_per_megapixel; self.update_status(f"Amélioration... Estimé : ~{int(self.estimated_total_time)}s")
        else: self.estimated_total_time = 0; self.update_status("Amélioration... (étalonnage en cours)")
        self.processing_start_time = time.time()
        thread = threading.Thread(target=self._process_image_worker, args=(pixels,))
        thread.daemon = True
        thread.start()
        self.after(100, self.update_progress)

    def _process_image_worker(self, pixels):
        start_time = time.time()
        try:
            restore_faces = self.cb_face_restore.get()
            # 1. Le worker traite l'image et récupère l'objet PIL
            result_pil_image = self.processor.process_image(self.input_path, restore_faces)

            # 2. Le worker PREPARE l'objet CTkImage, prêt à être affiché
            ctk_output_image = ctk.CTkImage(light_image=result_pil_image, size=(PREVIEW_SIZE[0], int(PREVIEW_SIZE[1] * result_pil_image.height / result_pil_image.width)))

            duration = time.time() - start_time
            self.progress_queue.put(('timing', (duration, pixels)))
            # 3. Le worker envoie l'objet PIL ET l'objet CTkImage à l'interface
            self.progress_queue.put(('done', (result_pil_image, ctk_output_image)))
        except Exception as e:
            self.progress_queue.put(('error', str(e)))

    def update_slider_label(self, value):
        self.color_slider_label.configure(text=f"Intensité : {int(value)}%")

    def apply_manual_color(self):
        if not self.output_pil_image: return
        alpha = self.color_slider.get() / 100.0
        if alpha == 0:
            self.displayed_output_image = self.output_pil_image
        else:
            fully_corrected_image = utils.correct_colors_grey_world(self.output_pil_image)
            if fully_corrected_image:
                self.displayed_output_image = Image.blend(self.output_pil_image, fully_corrected_image, alpha)
        self.update_output_preview()
        self.update_status(f"Correction manuelle appliquée à {int(alpha*100)}%.")

    def apply_auto_color(self):
        if not self.output_pil_image: return
        self.displayed_output_image = utils.correct_colors_grey_world(self.output_pil_image)
        self.update_output_preview()
        self.color_slider.set(100)
        self.update_slider_label(100)
        self.update_status("Correction automatique des couleurs appliquée.")

    def update_output_preview(self):
        """Met à jour uniquement l'aperçu de l'image améliorée, sans toucher aux autres images Tkinter."""
        if not self.displayed_output_image:
            return

        def _update_in_main_thread(pil_image):
            try:
                # Supprimer uniquement l'ancienne image de sortie, si elle existe
                if hasattr(self, "ctk_output_image") and self.ctk_output_image:
                    try:
                        del self.ctk_output_image
                    except Exception:
                        pass

                # Créer la nouvelle image CustomTkinter (toujours dans le thread principal)
                new_ctk_image = ctk.CTkImage(
                    light_image=pil_image,
                    size=(
                        PREVIEW_SIZE[0],
                        int(PREVIEW_SIZE[1] * pil_image.height / pil_image.width)
                    )
                )

                # Mettre à jour le label de sortie
                self.output_image_label.configure(image=new_ctk_image, text="")
                self.output_image_label.image = new_ctk_image
                self.ctk_output_image = new_ctk_image

                print("[INFO] Aperçu mis à jour avec succès.")
            except Exception as e:
                print(f"[ERREUR update_output_preview/_update_in_main_thread] {e}")

        # Exécuter dans le thread principal Tkinter
        self.after_idle(lambda: _update_in_main_thread(self.displayed_output_image))


    def update_progress(self):
        elapsed_time = time.time() - self.processing_start_time
        if self.estimated_total_time > 0:
            progress = min(elapsed_time / self.estimated_total_time, 0.99)
            self.progressbar.set(progress); self.time_label.configure(text=f"{int(elapsed_time)}s / ~{int(self.estimated_total_time)}s")
        else: self.time_label.configure(text=f"Temps écoulé : {int(elapsed_time)}s"); self.progressbar.set(elapsed_time / 60)
        try:
            message = self.progress_queue.get_nowait()
            msg_type, msg_data = message
            if msg_type == 'timing':
                duration, pixels = msg_data
                megapixels = pixels / 1_000_000
                if megapixels > 0: self.seconds_per_megapixel = duration / megapixels; self.save_config()
            elif msg_type == 'done':
                # 4. L'interface reçoit les deux objets
                pil_image, ctk_image = msg_data
                self.output_pil_image = pil_image
                self.displayed_output_image = pil_image

                # 5. L'interface n'a plus qu'à AFFICHER l'objet déjà prêt
                self.output_image_label.configure(image=ctk_image, text="")
                self.output_image_label.image = ctk_image # On ancre la référence par sécurité ultime

                self.update_status("Amélioration terminée !"); self.progressbar.set(1); self.time_label.configure(text=f"Terminé en {int(elapsed_time)}s")
                for btn in [self.btn_load, self.btn_process, self.btn_save, self.btn_auto_color, self.btn_apply_color]: btn.configure(state="normal")
                self.color_slider.configure(state="normal")
                return
            elif msg_type == 'error':
                messagebox.showerror("Erreur", str(msg_data)); self.update_status("Une erreur est survenue."); self.progressbar.set(0); self.time_label.configure(text="")
                for btn in [self.btn_load, self.btn_process]: btn.configure(state="normal")
                return
        except queue.Empty: pass
        self.after(100, self.update_progress)

    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.seconds_per_megapixel = config.get("seconds_per_megapixel", 0)
                if self.seconds_per_megapixel > 0: print(f"Vitesse de traitement chargée : {self.seconds_per_megapixel:.2f} s/Mpx")
        except (FileNotFoundError, json.JSONDecodeError): self.seconds_per_megapixel = 0

    def save_config(self):
        if self.seconds_per_megapixel <= 0: return
        config = {"seconds_per_megapixel": self.seconds_per_megapixel}
        with open(self.config_file, 'w') as f: json.dump(config, f, indent=4)
        print("Configuration sauvegardée.")

    def on_closing(self):
        self.save_config(); self.destroy()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path: return
        self.input_path = path
        self.output_pil_image = None; self.displayed_output_image = None
        self.output_image_label.configure(image=None, text="Le résultat apparaîtra ici.")
        for btn in [self.btn_save, self.btn_auto_color, self.btn_apply_color]: btn.configure(state="disabled")
        self.color_slider.configure(state="disabled"); self.color_slider.set(0); self.update_slider_label(0)
        self.progressbar.set(0); self.time_label.configure(text="")
        try:
            pil_image = Image.open(path)
            ctk_input_image = ctk.CTkImage(light_image=pil_image, size=(PREVIEW_SIZE[0], int(PREVIEW_SIZE[1] * pil_image.height / pil_image.width)))
            self.input_image_label.configure(image=ctk_input_image, text="")
            # On ancre l'image d'entrée pour être sûr
            self.input_image_label.image = ctk_input_image
            self.btn_process.configure(state="normal")
            self.update_status(f"Image chargée : {os.path.basename(path)}")
        except Exception as e: messagebox.showerror("Erreur d'ouverture", f"Impossible de charger l'image.\n{e}")

    def save_image(self):
        if not self.displayed_output_image: return
        input_dir, input_filename = os.path.split(self.input_path)
        name, ext = os.path.splitext(input_filename)
        default_name = f"{name}_enhanced.png"
        path = filedialog.asksaveasfilename(initialfile=default_name, defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            self.displayed_output_image.save(path)
            self.update_status(f"Image enregistrée sous : {os.path.basename(path)}")
            messagebox.showinfo("Succès", "L'image a été enregistrée avec succès.")

    def update_status(self, text):
        self.status_label.configure(text=text)
        self.update_idletasks()