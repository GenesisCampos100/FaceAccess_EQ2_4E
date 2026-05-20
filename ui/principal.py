"""
ui/principal.py — Clase principal FaceAccess
Orquesta los módulos core/, ui/, database/, models/ sin lógica propia.
"""

import cv2
import os
import sys
import threading
import time
import numpy as np
from queue import Queue
import customtkinter as ctk
from PIL import Image
from collections import Counter
from datetime import datetime

# Asegurar imports de módulos relativos
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.reconocimiento import cargar_encodings, buscar
from core.camara import CamaraManager
from ui.constantes import (
    C_BG, C_FRAME, C_FOOT, C_BORDE, C_OK, C_WARN, C_ERROR,
    C_TXT, C_TXT2, C_TXT3, C_ADMIN,
    H_HEADER, H_SALUDO, H_VIDEO, APP_GEOMETRY,
    HAAR_CASCADE, ESCALA_DETEC, MIN_VECINOS, MIN_TAMANO_RELAT,
    FRAMES_CONFIRM, PAUSA_SEG, FALLOS_NUMPAD, EVIDENCIAS_DIR, FOTOS_CAPTURA,
    KB_APP_W, KB_COLS, KB_PAD, KB_BH, KB_FS, KB_ACT_H,
)
from ui.teclado import TecladoVirtual
from database.db_manager import (
    obtener_usuario_por_id,
    obtener_usuario_por_matricula,
    registrar_entrada,
    registrar_intento_fallido,
    guardar_evidencia,
    guardar_encoding,
    login,
    puede_registrar,
    roles_asignables,
    registrar_usuario,
    DATA_DIR,
    ROL_ALUMNO,
)
from models.entrenadoRF import cargar_modelo_lbph, entrenar, FACE_SIZE


class FaceAccess(ctk.CTk):
    """
    Aplicación principal de reconocimiento facial con soporte para:
    - Acceso por reconocimiento facial
    - Captura de rostros (estudiante, personal, admin)
    - Registro manual con teclado virtual
    - Login con credenciales
    """

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("FaceAccess")
        self.geometry(APP_GEOMETRY)
        self.resizable(False, False)
        self.configure(fg_color=C_BG)

        # ── Estado de la aplicación ───────────────────────────────────────────
        self.estado = "escaneando"
        self.fallos = 0
        self._esperando_numpad = False
        self.cnt_in = 0
        self._pulso_fase = 0
        self._pulso_job = None
        self._ci = None
        self._np_val = ""
        self._np_vis = False

        # ── Locks y queue thread-safe ─────────────────────────────────────────
        self._lock = threading.Lock()
        self._lock_resultado = threading.Lock()
        self._lock_coords = threading.Lock()
        self._queue_frames = Queue(maxsize=2)

        # ── Variables internas de flujo ───────────────────────────────────────
        self._resultado = None
        self._res_nuevo = False
        self._ultimo_frame = None
        self._en_pausa = False
        self._t_pausa = None
        self._buffer = []
        self._frames_desc = 0
        self._ultimo_coords = None
        self._ultimo_id_u = None
        self._t_ultimo_res = 0.0
        self._buffer_varianza = []

        # ── Modelo LBPH ───────────────────────────────────────────────────────
        self._recognizer, _ = cargar_encodings()

        # ── Modos y flujos ────────────────────────────────────────────────────
        self._modo = "acceso"
        self._login_usuario = None
        self._login_validando = False
        self._login_frames_confirmados = 0
        self._login_frames_fallidos = 0
        self._cap_imagenes = []
        self._cap_count = 0
        self._reg_datos = {}
        self._coincidencias = []
        self._etapas_captura = []
        self._etapa_actual = 0
        self._foto_actual = 0
        self._rostro_detectado_frames = 0
        self._posicion_valida = False
        self._logo_clicks = 0
        self._logo_timer = None

        # ── Teclado táctil virtual ────────────────────────────────────────────
        self._teclado = TecladoVirtual(root=self)

        # ── Cámara ────────────────────────────────────────────────────────────
        self._camara = CamaraManager()

        # ── Construir UI ──────────────────────────────────────────────────────
        self._build_header()
        self._build_saludo()
        self._build_video()
        self._build_overlays()

        self._detector = self._init_haar()
        self._update_clock()
        self._pulso()
        self.protocol("WM_DELETE_WINDOW", self._cerrar)
        self.after(400, self._iniciar)

    def _init_haar(self):
        """Cargar detector Haar Cascade."""
        detector = cv2.CascadeClassifier(HAAR_CASCADE)
        if detector.empty():
            print(f"[ERROR] Haar Cascade no encontrado: {HAAR_CASCADE}")
            return None
        print("[MOTOR] Haar Cascade activo.")
        return detector

    def _build_header(self):
        """Construir barra de header."""
        frame = ctk.CTkFrame(self, fg_color=C_FRAME, corner_radius=0)
        frame.pack(fill="x", padx=0, pady=0)
        frame.configure(height=H_HEADER)
        frame.pack_propagate(False)
        ctk.CTkLabel(frame, text="🔐 FaceAccess", font=("Helvetica", 14, "bold"), text_color=C_TXT).pack(side="left", padx=15, pady=8)

    def _build_saludo(self):
        """Construir zona de saludo."""
        self.frame_saludo = ctk.CTkFrame(self, fg_color=C_BG, corner_radius=0)
        self.frame_saludo.pack(fill="x", padx=0, pady=0)
        self.frame_saludo.configure(height=H_SALUDO)
        self.frame_saludo.pack_propagate(False)
        self.lbl_saludo = ctk.CTkLabel(self.frame_saludo, text="Bienvenido", font=("Helvetica", 12), text_color=C_TXT2)
        self.lbl_saludo.pack(side="left", padx=15, pady=5)

    def _build_video(self):
        """Construir zona de video."""
        self.frame_video = ctk.CTkFrame(self, fg_color=C_BG, corner_radius=0)
        self.frame_video.pack(fill="both", expand=True, padx=0, pady=0)
        self.lbl_video = ctk.CTkLabel(self.frame_video, text="", bg_color=C_BG)
        self.lbl_video.pack(fill="both", expand=True, padx=5, pady=5)

    def _build_overlays(self):
        """Construir overlays (estado, debug)."""
        pass

    def _iniciar(self):
        """Iniciar cámara y loops."""
        self._camara.iniciar(lock=self._lock)
        self._t = threading.Thread(target=self._hilo_rec, daemon=True)
        self._t.start()
        self._t_cam = threading.Thread(target=self._loop_camara, daemon=True)
        self._t_cam.start()

    def _update_clock(self):
        """Actualizar reloj del sistema."""
        self.lbl_saludo.configure(text=datetime.now().strftime("%H:%M:%S"))
        self.after(1000, self._update_clock)

    def _pulso(self):
        """Pulsar LED/indicador (simulado)."""
        pass

    def _hilo_rec(self):
        """Hilo de reconocimiento facial (procesa buffer de frames)."""
        while True:
            try:
                if not self._queue_frames.empty():
                    frame_gray = self._queue_frames.get()
                    with self._lock_resultado:
                        if self._recognizer:
                            self._resultado = buscar(frame_gray, self._recognizer)
                            self._res_nuevo = True
                time.sleep(0.05)
            except Exception as e:
                print(f"[ERROR] Hilo reconocimiento: {e}")
                time.sleep(0.1)

    def _loop_camara(self):
        """Loop principal de cámara: detección + video."""
        while True:
            try:
                frame = self._camara.leer()
                if frame is None:
                    continue

                # Convertir a PIL y mostrar
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(rgb)
                photo = ctk.CTkImage(img_pil, size=(600, 600))
                self.lbl_video.configure(image=photo)
                self.lbl_video.image = photo

                time.sleep(0.033)
            except Exception as e:
                print(f"[ERROR] Loop cámara: {e}")
                time.sleep(0.1)

    def _cerrar(self):
        """Cerrar aplicación correctamente."""
        self._camara.liberar()
        self.destroy()


if __name__ == "__main__":
    app = FaceAccess()
    app.mainloop()
