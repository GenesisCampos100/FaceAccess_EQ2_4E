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
    H_HEADER, H_SALUDO, H_VIDEO, H_FOOTER, APP_GEOMETRY,
    HAAR_CASCADE, ESCALA_DETEC, MIN_VECINOS, MIN_TAMANO_RELAT,
    FRAMES_CONFIRM, PAUSA_SEG, FALLOS_NUMPAD, EVIDENCIAS_DIR, FOTOS_CAPTURA,
    KB_APP_W, KB_COLS, KB_PAD, KB_BH, KB_FS, KB_ACT_H,
)
from ui.constantes import s, sf, sw
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

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
ICONO_PATH = os.path.join(ASSETS_DIR, "icono.png")


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
        self.minsize(600, 1024)
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
        self._header_sep = ctk.CTkFrame(self, fg_color=C_BORDE, height=4, corner_radius=0)
        self._header_sep.pack(fill="x", pady=(0, 4))
        self._build_saludo()
        self._build_video()
        self._build_overlays()
        self._build_footer()

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
        """Construir barra de header con logo e información."""
        frame = ctk.CTkFrame(self, fg_color=C_FRAME, corner_radius=0, height=H_HEADER)
        frame.pack(fill="x")
        frame.pack_propagate(False)

        left = ctk.CTkFrame(frame, fg_color="transparent")
        left.pack(side="left", padx=18, pady=(4, 2), fill="y")

        logo_size = 38
        if os.path.exists(ICONO_PATH):
            try:
                logo_img = Image.open(ICONO_PATH)
                self._logo_icon = ctk.CTkImage(light_image=logo_img, dark_image=logo_img, size=(logo_size, logo_size))
                ctk.CTkLabel(left, text="", image=self._logo_icon, fg_color="transparent").pack(side="left", padx=(0, 12), anchor="center")
            except Exception as exc:
                print(f"[UI] No se pudo cargar el icono: {exc}")
                canvas = ctk.CTkCanvas(left, width=44, height=44, bg=C_FRAME, highlightthickness=0)
                canvas.pack(side="left", padx=(0, 12), anchor="center")
                canvas.create_oval(2, 2, 42, 42, fill=C_OK, outline="")
                canvas.create_text(22, 22, text="FA", fill=C_BG, font=("Helvetica", 14, "bold"))
        else:
            canvas = ctk.CTkCanvas(left, width=44, height=44, bg=C_FRAME, highlightthickness=0)
            canvas.pack(side="left", padx=(0, 12), anchor="center")
            canvas.create_oval(2, 2, 42, 42, fill=C_OK, outline="")
            canvas.create_text(22, 22, text="FA", fill=C_BG, font=("Helvetica", 14, "bold"))

        text_box = ctk.CTkFrame(left, fg_color="transparent")
        text_box.pack(side="left", anchor="center")
        ctk.CTkLabel(text_box, text="VisionID", font=("Helvetica", 17, "bold"), text_color=C_TXT).pack(anchor="w")
        ctk.CTkLabel(text_box, text="Control de Acceso", font=("Helvetica", 12), text_color=C_TXT2).pack(anchor="w")

        right = ctk.CTkFrame(frame, fg_color="transparent")
        right.pack(side="right", padx=18, pady=(4, 2), fill="y")
        self.lbl_hora = ctk.CTkLabel(right, text="", font=("Helvetica", 26, "bold"), text_color=C_TXT)
        self.lbl_hora.pack(anchor="e", pady=(10, 0))
        self.lbl_fecha = ctk.CTkLabel(right, text="", font=("Helvetica", 12), text_color=C_TXT2)
        self.lbl_fecha.pack(anchor="e")

    def _build_saludo(self):
        """Construir barra superior secundaria con estado y contadores."""
        self.frame_saludo = ctk.CTkFrame(self, fg_color=C_FOOT, corner_radius=0, height=H_SALUDO + 6)
        self.frame_saludo.pack(fill="x", pady=(1, 0))
        self.frame_saludo.pack_propagate(False)
        self.lbl_saludo = ctk.CTkLabel(self.frame_saludo, text="", font=("Helvetica", sf(13, self)), text_color=C_TXT2)
        self.lbl_saludo.pack(side="left", padx=18)
        right = ctk.CTkFrame(self.frame_saludo, fg_color="transparent")
        right.pack(side="right", padx=18)
        ctk.CTkLabel(right, text="●", font=("Helvetica", 9), text_color=C_OK).pack(side="left", padx=(0, 4))
        self.lbl_cnt_in = ctk.CTkLabel(right, text="0 entradas", font=("Helvetica", sf(13, self)), text_color=C_OK)
        self.lbl_cnt_in.pack(side="left")

    def _build_video(self):
        """Construir zona de video y overlays principales."""
        self.frame_video = ctk.CTkFrame(self, fg_color="#080F16", corner_radius=0, height=H_VIDEO)
        self.frame_video.pack(fill="both", expand=True)
        self.frame_video.pack_propagate(False)
        self.lbl_video = ctk.CTkLabel(self.frame_video, text="Iniciando cámara...", font=("Helvetica", 14), text_color=C_TXT2)
        self.lbl_video.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._badge_frame = ctk.CTkFrame(self.frame_video, fg_color=C_FRAME, corner_radius=10, bg_color="#080F16")
        self._badge_frame.place(relx=1.0, rely=0.0, anchor="ne", x=-12, y=12)
        self.lbl_badge = ctk.CTkLabel(self._badge_frame, text="● Escaneando", font=("Helvetica", 11, "bold"), text_color=C_OK, fg_color="transparent", corner_radius=0, padx=12, pady=4)
        self.lbl_badge.pack()
        self._inst_frame = ctk.CTkFrame(self.frame_video, fg_color="#0D1E2D", corner_radius=20, bg_color="#080F16")
        self.lbl_inst = ctk.CTkLabel(self._inst_frame, text="", font=("Helvetica", 14, "bold"), text_color=C_OK, fg_color="transparent", corner_radius=0, padx=20, pady=8)
        self.lbl_inst.pack()

    def _build_overlays(self):
        """Construir overlays secundarios."""
        self.ov_msg = ctk.CTkFrame(self.frame_video, corner_radius=0, fg_color="#0D1E2D", height=52)
        self.ov_msg.pack_propagate(False)
        self.lbl_msg_icono = ctk.CTkLabel(self.ov_msg, text="", font=("Helvetica", 20), fg_color="transparent", width=44)
        self.lbl_msg_icono.place(relx=0, rely=0.5, anchor="w", x=14)
        self._msg_texts = ctk.CTkFrame(self.ov_msg, fg_color="transparent")
        self._msg_texts.place(relx=0, rely=0.5, anchor="w", x=80)
        self.lbl_msg_titulo = ctk.CTkLabel(self._msg_texts, text="", font=("Helvetica", 13, "bold"), text_color=C_TXT, fg_color="transparent")
        self.lbl_msg_titulo.pack(anchor="w")
        self.lbl_msg_nombre = ctk.CTkLabel(self._msg_texts, text="", font=("Helvetica", 11), text_color=C_TXT2, fg_color="transparent")
        self.lbl_msg_nombre.pack(anchor="w")
        self.lbl_msg_info = ctk.CTkLabel(self.ov_msg, text="", font=("Helvetica", 10), text_color=C_TXT3, fg_color="transparent")
        self.lbl_msg_info.place(relx=1.0, rely=0.3, anchor="e", x=-16)
        self.prog_msg = ctk.CTkProgressBar(self.ov_msg, height=3, corner_radius=0, fg_color=C_BORDE, progress_color=C_OK)
        self.prog_msg.place(relx=0, rely=1.0, anchor="sw", relwidth=1)
        self.prog_msg.set(1.0)

        self._build_ov_registro()
        self._build_alerta_duplicado()
        self._build_ov_captura()

    def _build_footer(self):
        """Construir el footer con acción principal."""
        self.frame_footer = ctk.CTkFrame(self, fg_color=C_FOOT, corner_radius=0, height=H_FOOTER)
        self.frame_footer.pack(fill="x", side="bottom")
        self.frame_footer.pack_propagate(False)

        self.lbl_footer = ctk.CTkLabel(
            self.frame_footer,
            text="Control de acceso por reconocimiento facial",
            font=("Helvetica", 12),
            text_color=C_TXT2,
        )
        self.lbl_footer.pack(side="left", padx=18)

        self.btn_registrar = ctk.CTkButton(
            self.frame_footer,
            text="+ Registrar",
            width=150,
            height=40,
            fg_color=C_ADMIN,
            hover_color="#43379F",
            text_color=C_TXT,
            font=("Helvetica", 14, "bold"),
            corner_radius=10,
            command=self._abrir_registro,
        )
        self.btn_registrar.pack(side="right", padx=18, pady=16)

    def _build_ov_registro(self):
        """Construir el formulario de registro de usuario."""
        self.ov_registro = ctk.CTkFrame(self.frame_video, fg_color="#0F1923", corner_radius=0)
        ctk.CTkLabel(
            self.ov_registro, text="Registrar nuevo usuario",
            font=("Helvetica", sf(90, self), "bold"), text_color=C_TXT
        ).pack(pady=(s(12, self), s(6, self)))

        self.lbl_reg_op = ctk.CTkLabel(
            self.ov_registro, text="",
            font=("Helvetica", sf(20, self)), text_color=C_OK
        )
        self.lbl_reg_op.pack(pady=(0, s(8, self)))

        self._entries = {}
        form_grid = ctk.CTkFrame(self.ov_registro, fg_color="transparent")
        form_grid.pack(fill="both", expand=True, padx=s(20, self), pady=s(30, self))

        # Configurar rejilla de 2 columnas para que se expandan uniformemente
        form_grid.grid_columnconfigure(0, weight=1)
        form_grid.grid_columnconfigure(1, weight=1)

        campos = [
            ("Nombre(s)", "nombre"),
            ("Apellido paterno", "apellido_p"),
            ("Apellido materno", "apellido_m"),
            ("Matrícula", "matricula"),
            ("Contraseña", "contrasenia"),
            ("Confirmar contraseña", "contrasenia2"),
            ("Grado", "grado"),
            ("Grupo", "grupo"),
        ]
        self._grado_frame = None
        self._grupo_frame = None

        # Colocar cada campo en (row, col) según su índice (2 columnas)
        for i, (lbl, key) in enumerate(campos):
            row = i // 2
            col = i % 2
            f = ctk.CTkFrame(form_grid, fg_color="transparent")
            f.grid(row=row, column=col, padx=s(12, self), pady=s(12, self), sticky="ew")
            if key == "grado":
                self._grado_frame = f
                f.grid_remove()
            elif key == "grupo":
                self._grupo_frame = f
                f.grid_remove()

            ctk.CTkLabel(f, text=lbl, font=("Helvetica", sf(20, self)), text_color=C_TXT2).pack(anchor="w")

            if key in ("contrasenia", "contrasenia2"):
                fp = ctk.CTkFrame(f, fg_color="transparent")
                fp.pack(fill="x")
                e = ctk.CTkEntry(fp, width=sw(0.88, self), height=s(60, self), font=("Helvetica", sf(33, self)), show="*")
                e.pack(side="left", fill="x", expand=True)
                e.bind("<FocusIn>", lambda ev, entry=e: self._teclado.abrir(entry))
                vis = [False]

                def _toggle(en=e, v=vis):
                    v[0] = not v[0]
                    en.configure(show="" if v[0] else "*")

                ctk.CTkButton(
                    fp, text="👁", width=s(32, self), height=s(34, self),
                    fg_color=C_FRAME, hover_color=C_BORDE,
                    text_color=C_TXT2, font=("Helvetica", sf(20, self)),
                    command=_toggle
                ).pack(side="left", padx=(s(6, self), 0))
            else:
                e = ctk.CTkEntry(f, width=sw(0.92, self), height=s(60, self), font=("Helvetica", sf(33, self)))
                e.pack(fill="x", expand=True)
                e.bind("<FocusIn>", lambda ev, entry=e: self._teclado.abrir(entry))

            self._entries[key] = e

        rol_frame = ctk.CTkFrame(form_grid, fg_color="transparent")
        rol_frame.grid(row=len(campos), column=0, columnspan=2, padx=s(12, self), pady=s(8, self), sticky="w")
        ctk.CTkLabel(rol_frame, text="Rol", font=("Helvetica", sf(20, self)), text_color=C_TXT2).pack(anchor="w")
        self.combo_rol = ctk.CTkComboBox(
            rol_frame,
            width=sw(0.8, self),
            height=s(48, self),
            font=("Helvetica", sf(33, self)),
            values=["ALUMNO", "PERSONAL_ESCOLAR"],
            command=self._actualizar_campos_rol,
        )
        self.combo_rol.pack()
        self.combo_rol.set("ALUMNO")

        self.lbl_reg_err = ctk.CTkLabel(self.ov_registro, text="", font=("Helvetica", sf(21, self)), text_color=C_ERROR)
        self.lbl_reg_err.pack(pady=s(4, self))

        fb = ctk.CTkFrame(self.ov_registro, fg_color="transparent")
        fb.pack(pady=s(12, self))
        ctk.CTkButton(
            fb, text="Continuar →", width=sw(0.5, self), height=s(56, self), fg_color=C_OK,
            text_color=C_BG, hover_color="#00A88A",
            font=("Helvetica", sf(33, self), "bold"),
            command=self._reg_continuar
        ).pack(side="left", padx=s(6, self))

    def _build_alerta_duplicado(self):
        """Construir alerta de rostro duplicado."""
        self.ov_duplicado = ctk.CTkFrame(self.frame_video, corner_radius=0, fg_color="#0D1E2D")
        self.ov_duplicado.pack_propagate(False)
        self._dup_inner = ctk.CTkFrame(self.ov_duplicado, fg_color="transparent")
        self._dup_inner.place(relx=0.5, rely=0.5, anchor="center")
        self.lbl_dup_icono = ctk.CTkLabel(self._dup_inner, text="⛔", font=("Helvetica", 52), fg_color="transparent")
        self.lbl_dup_icono.pack(pady=(0, 8))
        self.lbl_dup_titulo = ctk.CTkLabel(self._dup_inner, text="ROSTRO DUPLICADO", font=("Helvetica", 18, "bold"), text_color=C_ERROR, fg_color="transparent")
        self.lbl_dup_titulo.pack(pady=(0, 8))
        self.lbl_dup_nombre = ctk.CTkLabel(self._dup_inner, text="", font=("Helvetica", 13, "bold"), text_color=C_TXT, fg_color="transparent")
        self.lbl_dup_nombre.pack(pady=(0, 8))
        self.lbl_dup_msg = ctk.CTkLabel(self._dup_inner, text="Este rostro ya está registrado", font=("Helvetica", 11), text_color=C_TXT2, fg_color="transparent", justify="center")
        self.lbl_dup_msg.pack(pady=(0, 16))
        self.prog_dup = ctk.CTkProgressBar(self._dup_inner, width=200, height=4, corner_radius=2, fg_color=C_BORDE, progress_color=C_ERROR)
        self.prog_dup.pack()
        self.prog_dup.set(1.0)

    def _mostrar_alerta_duplicado(self, nombre_usuario):
        self.lbl_dup_nombre.configure(text=nombre_usuario)
        self.prog_dup.set(1.0)
        self.ov_duplicado.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.ov_duplicado.lift()

    def _ocultar_alerta_duplicado(self):
        if hasattr(self, "ov_duplicado"):
            self.ov_duplicado.place_forget()

    def _ocultar_msg(self):
        if hasattr(self, "ov_msg"):
            self.ov_msg.place_forget()

    def _detectar(self, gray_small):
        """Detectar el rostro más grande en un frame reducido."""
        if not hasattr(self, "_detector") or self._detector is None:
            return None
        h_sm, w_sm = gray_small.shape[:2]
        min_size = (
            max(int(w_sm * MIN_TAMANO_RELAT), 20),
            max(int(h_sm * MIN_TAMANO_RELAT), 20),
        )
        rostros = self._detector.detectMultiScale(
            gray_small,
            scaleFactor=1.1,
            minNeighbors=MIN_VECINOS,
            minSize=min_size,
        )
        if len(rostros) == 0:
            return None
        return max(rostros, key=lambda r: r[2] * r[3])

    def _build_ov_captura(self):
        """Construir el overlay de captura de rostro."""
        self.ov_captura = ctk.CTkFrame(self.frame_video, fg_color="#080F16", corner_radius=0)
        ctk.CTkLabel(self.ov_captura, text="Captura de rostro", font=("Helvetica", 15, "bold"), text_color=C_TXT).pack(pady=(24, 2))
        self.lbl_cap_nombre = ctk.CTkLabel(self.ov_captura, text="", font=("Helvetica", 12), text_color=C_OK)
        self.lbl_cap_nombre.pack(pady=(0, 10))
        self.lbl_cap_video = ctk.CTkLabel(self.ov_captura, text="", width=320, height=240)
        self.lbl_cap_video.pack()
        ctk.CTkLabel(self.ov_captura, text="Mueve la cabeza en distintos ángulos", font=("Helvetica", 11), text_color=C_TXT2).pack(pady=8)
        self.prog_cap = ctk.CTkProgressBar(self.ov_captura, width=320, height=6, corner_radius=3, fg_color=C_BORDE, progress_color=C_OK)
        self.prog_cap.pack(); self.prog_cap.set(0)
        self.lbl_cap_cnt = ctk.CTkLabel(self.ov_captura, text=f"0 / {FOTOS_CAPTURA} fotos", font=("Helvetica", 12), text_color=C_TXT2)
        self.lbl_cap_cnt.pack(pady=6)
        self.lbl_cap_instruc = ctk.CTkLabel(self.ov_captura, text="", font=("Helvetica", 11), text_color=C_WARN)
        self.lbl_cap_instruc.pack(pady=(0, 4))
        self.lbl_cap_estado = ctk.CTkLabel(self.ov_captura, text="", font=("Helvetica", 12, "bold"), text_color=C_OK)
        self.lbl_cap_estado.pack(pady=2)
        ctk.CTkButton(self.ov_captura, text="Cancelar", fg_color="transparent", text_color=C_TXT2, hover_color=C_FRAME, font=("Helvetica", 12), command=lambda: self._confirmar_cancelar("¿Cancelar la captura? Se perderán las fotos tomadas.")).pack(pady=(10, 0))

    def _abrir_captura(self):
        self._modo = "captura"
        self._cap_imagenes = []
        self._coincidencias = []
        self._cap_count = 0
        self._etapas_captura = [
            {"nombre": "Frente", "mensaje": "Mira al frente", "fotos": 10},
            {"nombre": "Izquierda", "mensaje": "Gira ligeramente a la izquierda", "fotos": 10},
            {"nombre": "Derecha", "mensaje": "Gira ligeramente a la derecha", "fotos": 10},
        ]
        self._etapa_actual = 0
        self._foto_actual = 0
        self._rostro_detectado_frames = 0
        self._posicion_valida = False
        nom = f"{self._reg_datos['nombre']} {self._reg_datos['apellido_p']}"
        self.lbl_cap_nombre.configure(text=nom)
        self.prog_cap.set(0)
        self._actualizar_indicacion_captura()
        self._ocultar_overlays()
        self.ov_captura.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._loop_captura()

    def _actualizar_indicacion_captura(self):
        if not self._etapas_captura:
            return
        etapa = self._etapas_captura[self._etapa_actual]
        self.lbl_cap_estado.configure(text=etapa["mensaje"], text_color=C_OK)
        self.lbl_cap_cnt.configure(text=f"{self._cap_count} / {FOTOS_CAPTURA} fotos  |  [{self._foto_actual+1}/{etapa['fotos']}]")

    def _loop_captura(self):
        if self._modo != "captura" or self._cap_count >= FOTOS_CAPTURA:
            return
        frame = self._camara.leer()
        if frame is None:
            self.after(150, self._loop_captura)
            return
        try:
            h_orig, w_orig = frame.shape[:2]
            target_w = 360
            target_h = int(360 * h_orig / w_orig)
            if target_h > 390:
                target_h = 390
                target_w = int(390 * w_orig / h_orig)
            fd = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((390, 360, 3), dtype=np.uint8)
            canvas.fill(10)
            y_off = (390 - target_h) // 2
            x_off = (360 - target_w) // 2
            canvas[y_off:y_off+target_h, x_off:x_off+target_w] = fd
            marco_w = int(target_w * 0.60)
            marco_h = int(target_h * 0.70)
            marco_x = x_off + (target_w - marco_w) // 2
            marco_y = y_off + (target_h - marco_h) // 2
            color_marco = (0, 212, 170) if self._posicion_valida else (245, 166, 35)
            cv2.rectangle(canvas, (marco_x, marco_y), (marco_x + marco_w, marco_y + marco_h), color_marco, 3)
            img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            ci = ctk.CTkImage(light_image=img, dark_image=img, size=(360, 390))
            self.lbl_cap_video.configure(image=ci, text="")
            self._ci_cap = ci
        except Exception as e:
            print(f"[ERROR CAPTURA] {e}")
            self.after(80, self._loop_captura)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (0, 0), fx=ESCALA_DETEC, fy=ESCALA_DETEC)
        coords = self._detectar(small)
        rostro_bien_posicionado = False
        instruccion = ""

        if coords is not None:
            x, y, w, h = coords
            h_sm, w_sm = small.shape[:2]
            ratio_altura = h / h_sm
            if ratio_altura < 0.15:
                instruccion = "Acércate más"
            elif ratio_altura > 0.70:
                instruccion = "Aléjate un poco"
            else:
                centro_x = (x + w / 2) / w_sm
                centro_y = (y + h / 2) / h_sm
                if abs(centro_x - 0.5) > 0.15:
                    instruccion = "Centra tu rostro"
                elif abs(centro_y - 0.45) > 0.15:
                    instruccion = "Ajusta la altura"
                else:
                    rostro_bien_posicionado = True
                    instruccion = "✓ Posición correcta"
            self._posicion_valida = rostro_bien_posicionado
            self._rostro_detectado_frames = self._rostro_detectado_frames + 1 if rostro_bien_posicionado else 0
            margen = 10
            x1 = max(x + margen, 0)
            y1 = max(y + margen, 0)
            x2 = min(x + w - margen, small.shape[1])
            y2 = min(y + h - margen, small.shape[0])
            rostro_crop = small[y1:y2, x1:x2]

            if rostro_crop.size > 0:
                rostro_res = cv2.resize(rostro_crop, FACE_SIZE)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                rostro_res = clahe.apply(rostro_res)
                rostro_res = cv2.bilateralFilter(rostro_res, 5, 75, 75)
                rostro_res = cv2.normalize(rostro_res, None, 0, 255, cv2.NORM_MINMAX)

                if self._recognizer is not None and self._cap_count >= 15 and rostro_bien_posicionado:
                    try:
                        id_existente, confianza = buscar(rostro_crop, self._recognizer)
                        if id_existente is not None and confianza < 55:
                            self._coincidencias.append(id_existente)
                        if len(self._coincidencias) >= 5:
                            id_rep = max(set(self._coincidencias), key=self._coincidencias.count)
                            if self._coincidencias.count(id_rep) >= 3:
                                u_dup = obtener_usuario_por_id(id_rep)
                                nombre_dup = f"{u_dup['nombre']} {u_dup['apellido_p']}" if u_dup else "Usuario existente"
                                self._mostrar_alerta_duplicado(nombre_dup)
                                self._modo = "cerrado_captura"
                                self.after(4000, self._volver_a_formulario_registro)
                                return
                    except Exception as e:
                        print(f"[VALIDACIÓN] {e}")

                if self._rostro_detectado_frames >= 3:
                    self._cap_imagenes.append(rostro_res)
                    self._foto_actual += 1
                    self._cap_count += 1
                    self._rostro_detectado_frames = 0
                    etapa = self._etapas_captura[self._etapa_actual]
                    self.prog_cap.set(self._cap_count / FOTOS_CAPTURA)
                    self.lbl_cap_cnt.configure(text=f"{self._cap_count} / {FOTOS_CAPTURA} fotos  |  [{self._foto_actual}/{etapa['fotos']}]")
                    if self._foto_actual >= etapa["fotos"]:
                        self._etapa_actual += 1
                        self._foto_actual = 0
                        if self._etapa_actual >= len(self._etapas_captura):
                            self.after(150, self._finalizar_registro)
                            return
                        self.lbl_cap_instruc.configure(text="Preparando siguiente posición...", text_color=C_WARN)
                        self.lbl_cap_estado.configure(text=self._etapas_captura[self._etapa_actual]["mensaje"], text_color=C_OK)
                        self.after(2000, self._loop_captura)
                        return
                    self.after(600, self._loop_captura)
                    return
        else:
            self._rostro_detectado_frames = 0
            self._posicion_valida = False
            instruccion = "Acerca tu rostro"

        etapa = self._etapas_captura[self._etapa_actual]
        estado_color = C_OK if self._posicion_valida else C_WARN
        self.lbl_cap_estado.configure(text=etapa["mensaje"], text_color=estado_color)
        self.lbl_cap_instruc.configure(text=instruccion, text_color=estado_color)
        self.after(150, self._loop_captura)

    def _volver_a_formulario_registro(self):
        """Volver al formulario si se detecta un duplicado."""
        self._cap_imagenes = []
        self._cap_count = 0
        self._coincidencias = []
        self._etapa_actual = 0
        self._foto_actual = 0
        self._rostro_detectado_frames = 0
        self.prog_cap.set(0)
        self.lbl_cap_cnt.configure(text=f"0 / {FOTOS_CAPTURA} fotos")
        self._ocultar_alerta_duplicado()
        self.ov_captura.place_forget()
        self._modo = "registro"
        self.lbl_reg_err.configure(text="⚠ Rostro duplicado. Verifica los datos.", text_color=C_ERROR)
        self.ov_registro.place(relx=0, rely=0, relwidth=1, relheight=1)

    def _validar_rostro_duplicado(self):
        if not self._cap_imagenes or self._recognizer is None:
            return False, None, 999.0
        confianza_minima = 999.0
        id_duplicado = None
        for img_gray in self._cap_imagenes:
            rostro_res = cv2.resize(img_gray, FACE_SIZE)
            label, confianza = self._recognizer.predict(rostro_res)
            if confianza < 55 and confianza < confianza_minima:
                confianza_minima = confianza
                id_duplicado = label
        return confianza_minima < 55, id_duplicado, confianza_minima

    def _mostrar_exito_registro(self, nom_reg):
        self.ov_exito_reg = ctk.CTkFrame(self.frame_video, fg_color="#080F16", corner_radius=0)
        self.ov_exito_reg.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.ov_exito_reg.lift()
        inner = ctk.CTkFrame(self.ov_exito_reg, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(inner, text="✅", font=("Helvetica", 52), fg_color="transparent").pack(pady=(0, 10))
        ctk.CTkLabel(inner, text="¡Registro exitoso!", font=("Helvetica", 20, "bold"), text_color=C_OK, fg_color="transparent").pack(pady=(0, 6))
        ctk.CTkLabel(inner, text=nom_reg, font=("Helvetica", 14), text_color=C_TXT, fg_color="transparent").pack(pady=(0, 4))
        ctk.CTkLabel(inner, text="El modelo se está actualizando en segundo plano.", font=("Helvetica", 11), text_color=C_TXT2, fg_color="transparent").pack(pady=(0, 24))
        ctk.CTkButton(inner, text="Listo  ✓", width=200, height=46, fg_color=C_OK, text_color=C_BG, hover_color="#00A88A", font=("Helvetica", 15, "bold"), corner_radius=12, command=self._listo_registro).pack()

    def _listo_registro(self):
        if hasattr(self, "ov_exito_reg") and self.ov_exito_reg.winfo_exists():
            self.ov_exito_reg.place_forget()
            self.ov_exito_reg.destroy()
        self._ocultar_overlays()
        self._modo = "acceso"
        self.estado = "escaneando"
        self._reg_datos = {}
        self._cap_imagenes = []
        self._coincidencias = []
        self._etapa_actual = 0
        self._foto_actual = 0
        self._rostro_detectado_frames = 0
        self.prog_cap.set(0)
        self.lbl_cap_cnt.configure(text=f"0 / {FOTOS_CAPTURA} fotos")
        self.lbl_reg_err.configure(text="")
        for e in self._entries.values():
            e.delete(0, "end")
        self.combo_rol.set("ALUMNO")
        self._actualizar_campos_rol("ALUMNO")

    def _finalizar_registro(self):
        self.lbl_cap_estado.configure(text="Procesando...", text_color=C_WARN)
        self.update()
        try:
            es_dup, id_dup, _ = self._validar_rostro_duplicado()
            if es_dup:
                u_dup = obtener_usuario_por_id(id_dup)
                nom_dup = f"{u_dup['nombre']} {u_dup['apellido_p']}" if u_dup else f"Usuario ID {id_dup}"
                self.lbl_cap_estado.configure(text=f"✗ Rostro duplicado: {nom_dup}.", text_color=C_ERROR)
                self.after(3000, self._volver_a_formulario_registro)
                return

            id_u = registrar_usuario(
                nombre=self._reg_datos["nombre"],
                apellido_p=self._reg_datos["apellido_p"],
                matricula=self._reg_datos["matricula"],
                contrasenia=self._reg_datos["contrasenia"],
                id_rol=self._reg_datos["id_rol"],
                apellido_m=self._reg_datos.get("apellido_m", ""),
                grado=self._reg_datos.get("grado", ""),
                grupo=self._reg_datos.get("grupo", ""),
            )
            if not id_u:
                raise Exception("No se pudo obtener ID del nuevo usuario")

            nom_carpeta = f"{self._reg_datos['nombre']}_{self._reg_datos['apellido_p']}"
            carpeta = os.path.join(DATA_DIR, f"{id_u}_{nom_carpeta}")
            os.makedirs(carpeta, exist_ok=True)
            for idx, img in enumerate(self._cap_imagenes):
                ruta_img = os.path.join(carpeta, f"rostro_{idx:03d}.jpg")
                if not cv2.imwrite(ruta_img, img):
                    raise Exception(f"Error escribiendo imagen {idx}")

            guardar_encoding(id_u, carpeta)
            nom_reg = f"{self._reg_datos['nombre']} {self._reg_datos['apellido_p']}"

            def _reentrenar():
                try:
                    ok = entrenar()
                    if ok:
                        nuevo_rec = cargar_modelo_lbph()
                        with self._lock:
                            self._recognizer = nuevo_rec
                except Exception as e:
                    print(f"[ERROR] Re-entrenamiento: {e}")

            threading.Thread(target=_reentrenar, daemon=True).start()
            self._mostrar_exito_registro(nom_reg)

        except Exception as e:
            print(f"[ERROR REGISTRO] {e}")
            self.lbl_cap_estado.configure(text=f"Error: {e}", text_color=C_ERROR)

        ctk.CTkButton(
            fb, text="Cancelar", width=100, height=40,
            fg_color="transparent", text_color=C_TXT2,
            hover_color=C_FRAME, font=("Helvetica", 12),
            command=lambda: self._confirmar_cancelar("¿Cancelar el registro? Se perderán los datos ingresados.")
        ).pack(side="left", padx=6)

    def _actualizar_campos_rol(self, rol_seleccionado):
        """Mostrar u ocultar campos según el rol."""
        if rol_seleccionado == "ALUMNO":
            if self._grado_frame:
                self._grado_frame.grid()
            if self._grupo_frame:
                self._grupo_frame.grid()
        else:
            if self._grado_frame:
                self._grado_frame.grid_remove()
            if self._grupo_frame:
                self._grupo_frame.grid_remove()

    def _reg_continuar(self):
        """Validar el formulario de registro y continuar a captura."""
        datos = {k: e.get().strip() for k, e in self._entries.items()}
        mapa = {"ADMIN": 1, "PERSONAL_AUTORIZADO": 2, "PERSONAL_ESCOLAR": 3, "ALUMNO": 4}
        rol_sel = self.combo_rol.get()

        if not all([datos["nombre"], datos["apellido_p"], datos["matricula"], datos["contrasenia"]]):
            self.lbl_reg_err.configure(text="Nombre, apellido, matrícula y contraseña son obligatorios.")
            return

        if rol_sel == "ALUMNO":
            if not datos.get("grado") or not datos.get("grupo"):
                self.lbl_reg_err.configure(text="Grado y grupo son obligatorios para alumnos.")
                return
            if not datos["grado"].isdigit():
                self.lbl_reg_err.configure(text="El grado solo debe ser un número (ej: 1, 2, 10).")
                return
            if not datos["grupo"].isalpha() or len(datos["grupo"]) != 1:
                self.lbl_reg_err.configure(text="El grupo solo debe ser una letra (A, B, C...).")
                return
            if not (1 <= int(datos["grado"]) <= 12):
                self.lbl_reg_err.configure(text="El grado debe estar entre 1 y 12.")
                return

        if not datos["nombre"].replace(" ", "").isalpha():
            self.lbl_reg_err.configure(text="El nombre solo debe contener letras.")
            return
        if not datos["apellido_p"].replace(" ", "").isalpha():
            self.lbl_reg_err.configure(text="El apellido paterno solo debe contener letras.")
            return
        if datos["apellido_m"] and not datos["apellido_m"].replace(" ", "").isalpha():
            self.lbl_reg_err.configure(text="El apellido materno solo debe contener letras.")
            return
        if obtener_usuario_por_matricula(datos["matricula"].upper()):
            self.lbl_reg_err.configure(text=f"La matrícula '{datos['matricula'].upper()}' ya existe.")
            return
        if len(datos["contrasenia"]) < 6:
            self.lbl_reg_err.configure(text="La contraseña debe tener mínimo 6 caracteres.")
            return
        if datos["contrasenia"] != datos.get("contrasenia2", ""):
            self.lbl_reg_err.configure(text="Las contraseñas no coinciden.")
            return

        datos["nombre"] = datos["nombre"].upper()
        datos["apellido_p"] = datos["apellido_p"].upper()
        datos["apellido_m"] = datos["apellido_m"].upper() if datos["apellido_m"] else ""
        datos["matricula"] = datos["matricula"].upper()
        datos["grado"] = datos.get("grado", "").upper()
        datos["grupo"] = datos.get("grupo", "").upper()
        datos["id_rol"] = mapa.get(rol_sel, 4)
        self._reg_datos = datos
        self._abrir_captura()

    def _confirmar_cancelar(self, mensaje, accion_si=None):
        """Cancelar el overlay activo y volver al escaneo."""
        self._ocultar_overlays()
        self._ocultar_msg()
        self._ocultar_alerta_duplicado()
        self.ov_registro.place_forget()
        self.ov_captura.place_forget()
        self._inst_frame.place_forget()
        self._modo = "acceso"
        self.estado = "escaneando"
        self.lbl_badge.configure(text="● Escaneando", text_color=C_OK)
        self._set_estado = getattr(self, "_set_estado", None)
        if callable(accion_si):
            accion_si()

    def _ocultar_overlays(self):
        """Ocultar overlays superpuestos."""
        for name in ("ov_msg", "ov_duplicado", "ov_numpad", "ov_login", "ov_registro", "ov_captura"):
            widget = getattr(self, name, None)
            if widget is not None:
                try:
                    widget.place_forget()
                except Exception:
                    pass

    def _abrir_registro(self, operador=None):
        """Abrir el formulario de registro desde el footer."""
        self._modo = "registro"
        self.estado = "registro"
        self._ocultar_overlays()

        if operador and hasattr(self, "combo_rol"):
            mapa = {1: "ADMIN", 2: "PERSONAL_AUTORIZADO", 3: "PERSONAL_ESCOLAR", 4: "ALUMNO"}
            opciones = [mapa[r] for r in roles_asignables(operador["id_rol"]) if r in mapa]
            if opciones:
                self.combo_rol.configure(values=opciones)
                rol_inicial = opciones[-1]
                self.combo_rol.set(rol_inicial)
                self._actualizar_campos_rol(rol_inicial)
                self.lbl_reg_op.configure(text=f"Operador: {operador['nombre']} {operador['apellido_p']} ({operador['nombre_rol']})")
        else:
            if hasattr(self, "combo_rol"):
                self.combo_rol.configure(values=["ALUMNO", "PERSONAL_ESCOLAR"])
                self.combo_rol.set("ALUMNO")
                self._actualizar_campos_rol("ALUMNO")
            if hasattr(self, "lbl_reg_op"):
                self.lbl_reg_op.configure(text="Completa los datos para registrar un nuevo usuario")

        if hasattr(self, "lbl_badge"):
            self.lbl_badge.configure(text="● Registro", text_color=C_OK)
        self.ov_registro.place(relx=0, rely=0, relwidth=1, relheight=1)

    def _iniciar(self):
        """Iniciar cámara y loops."""
        self._camara.iniciar(lock=self._lock)
        self._t = threading.Thread(target=self._hilo_rec, daemon=True)
        self._t.start()
        self._t_cam = threading.Thread(target=self._loop_camara, daemon=True)
        self._t_cam.start()

    def _update_clock(self):
        """Actualizar reloj y fecha del sistema."""
        ahora = datetime.now()
        self.lbl_hora.configure(text=ahora.strftime("%H:%M"))
        self.lbl_fecha.configure(text=ahora.strftime("%d/%m/%Y"))
        self.lbl_saludo.configure(text="Buenos días ☼")
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

                # Mantener proporción para evitar recortes o distorsión
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(rgb)
                photo = ctk.CTkImage(img_pil, size=(600, H_VIDEO))
                self.lbl_video.configure(image=photo, text="")
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
