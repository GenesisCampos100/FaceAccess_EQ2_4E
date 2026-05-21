"""
ui/admin_setup.py — Setup inicial del primer administrador
Corre UNA sola vez desde main.py cuando la BD está vacía de admins.
"""

import cv2
import os
import sys
import numpy as np
import customtkinter as ctk
from PIL import Image

# Asegurar imports de módulos relativos
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import (
    get_connection, guardar_encoding, obtener_usuario_por_matricula,
    DATA_DIR, ROL_ADMIN
)
from models.entrenadoRF import entrenar, FACE_SIZE
from core.camara import CamaraManager
from ui.constantes import (
    C_BG, C_FRAME, C_BORDE, C_OK, C_WARN, C_ERROR, C_TXT, C_TXT2,
    APP_GEOMETRY, HAAR_CASCADE, ESCALA_DETEC, MIN_VECINOS, MIN_TAMANO_RELAT,
    H_HEADER, H_VIDEO, s, sf, sw
)
from ui.teclado import TecladoVirtual

FOTOS_CAPTURA = 30
ETAPAS = [
    {"mensaje": "Mira al frente", "fotos": 10},
    {"mensaje": "Gira ligeramente a la izquierda", "fotos": 10},
    {"mensaje": "Gira ligeramente a la derecha", "fotos": 10},
]


class AdminSetup(ctk.CTk):
    """Asistente de configuración para primer administrador."""

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("VisionID — Setup inicial")
        self.geometry(APP_GEOMETRY)
        self.resizable(False, False)
        self.configure(fg_color=C_BG)
        self.protocol("WM_DELETE_WINDOW", self._cerrar)

        self._camara = CamaraManager()
        self._teclado = TecladoVirtual(root=self)
        det = cv2.CascadeClassifier(HAAR_CASCADE)
        self._detector = None if det.empty() else det

        self._cap_imagenes = []
        self._cap_count = 0
        self._etapa_actual = 0
        self._foto_actual = 0
        self._rostro_detectado_frames = 0
        self._posicion_valida = False
        self._modo = "formulario"
        self._ci_cap = None

        self._build_header()
        self._build_formulario()
        self._build_captura()
        self.after(400, lambda: self._camara.iniciar())

    def _build_header(self):
        """Construir barra de header."""
        f = ctk.CTkFrame(self, fg_color=C_FRAME, corner_radius=0, height=s(H_HEADER, self))
        f.pack(fill="x")
        f.pack_propagate(False)
        ctk.CTkLabel(
            f, text="⚙  Setup inicial — Primer administrador",
            font=("Helvetica", sf(15, self), "bold"), text_color=C_WARN
        ).pack(side="left", padx=20, pady=18)

    def _build_formulario(self):
        """Construir formulario de datos del admin."""
        self.frm_form = ctk.CTkFrame(self, fg_color=C_BG)
        self.frm_form.pack(fill="both", expand=True)

        ctk.CTkLabel(
            self.frm_form, text="Ingresa los datos del administrador principal",
            font=("Helvetica", sf(13, self)), text_color=C_TXT2
        ).pack(pady=(s(20, self), s(10, self)))

        grid = ctk.CTkFrame(self.frm_form, fg_color="transparent")
        grid.place(relx=0.5, rely=0.5, anchor="center")

        campos = [
            ("Nombre(s)", "nombre"),
            ("Apellido paterno", "apellido_p"),
            ("Apellido materno", "apellido_m"),
            ("Matrícula", "matricula"),
            ("Contraseña", "contrasenia"),
            ("Confirmar contraseña", "contrasenia2"),
        ]
        self._entries = {}
        for i, (lbl, key) in enumerate(campos):
            f = ctk.CTkFrame(grid, fg_color="transparent")
            f.grid(row=i, column=0, columnspan=2, padx=s(12, self), pady=s(8, self), sticky="w")
            ctk.CTkLabel(f, text=lbl, font=("Helvetica", sf(16, self)), text_color=C_TXT2).pack(anchor="w")

            if "contrasenia" in key:
                fp = ctk.CTkFrame(f, fg_color="transparent")
                fp.pack(anchor="w")
                e = ctk.CTkEntry(fp, width=sw(0.8, self), height=s(48, self), font=("Helvetica", sf(15, self)), show="*")
                e.pack(side="left")
                e.bind("<FocusIn>", lambda ev, en=e: self._teclado.abrir(en))
                vis = [False]

                def _toggle(en=e, v=vis):
                    v[0] = not v[0]
                    en.configure(show="" if v[0] else "*")

                ctk.CTkButton(
                    fp, text="👁", width=s(30, self), height=s(36, self),
                    fg_color=C_FRAME, hover_color=C_BORDE,
                    text_color=C_TXT2, font=("Helvetica", sf(14, self)),
                    command=_toggle
                ).pack(side="left", padx=(s(2, self), 0))
            else:
                e = ctk.CTkEntry(f, width=sw(0.8, self), height=s(48, self), font=("Helvetica", sf(15, self)))
                e.pack()
                e.bind("<FocusIn>", lambda ev, en=e: self._teclado.abrir(en))

            self._entries[key] = e

        self.lbl_err = ctk.CTkLabel(
            self.frm_form, text="",
            font=("Helvetica", sf(11, self)), text_color=C_ERROR
        )
        self.lbl_err.pack(pady=s(6, self))

        ctk.CTkButton(
            self.frm_form, text="Continuar →", width=s(200, self), height=s(44, self),
            fg_color=C_OK, text_color=C_BG, hover_color="#00A88A",
            font=("Helvetica", sf(14, self), "bold"), corner_radius=s(12, self),
            command=self._validar_form
        ).pack(pady=s(8, self))

    def _validar_form(self):
        """Validar datos del formulario."""
        d = {k: e.get().strip() for k, e in self._entries.items()}

        if not all([d["nombre"], d["apellido_p"], d["matricula"], d["contrasenia"]]):
            self.lbl_err.configure(text="Nombre, apellido, matrícula y contraseña son obligatorios.")
            return

        if not d["nombre"].replace(" ", "").isalpha():
            self.lbl_err.configure(text="El nombre solo debe contener letras.")
            return

        if not d["apellido_p"].replace(" ", "").isalpha():
            self.lbl_err.configure(text="El apellido paterno solo debe contener letras.")
            return

        if d["apellido_m"] and not d["apellido_m"].replace(" ", "").isalpha():
            self.lbl_err.configure(text="El apellido materno solo debe contener letras.")
            return

        if obtener_usuario_por_matricula(d["matricula"].upper()):
            self.lbl_err.configure(text=f"La matrícula '{d['matricula'].upper()}' ya existe.")
            return

        if len(d["contrasenia"]) < 6:
            self.lbl_err.configure(text="La contraseña debe tener mínimo 6 caracteres.")
            return

        if d["contrasenia"] != d["contrasenia2"]:
            self.lbl_err.configure(text="Las contraseñas no coinciden.")
            return

        self._reg_datos = {
            "nombre": d["nombre"].upper(),
            "apellido_p": d["apellido_p"].upper(),
            "apellido_m": d["apellido_m"].upper() if d["apellido_m"] else "",
            "matricula": d["matricula"].upper(),
            "contrasenia": d["contrasenia"],
        }
        self._abrir_captura()

    def _build_captura(self):
        """Construir UI de captura de rostro."""
        self.frm_cap = ctk.CTkFrame(self, fg_color=C_BG)
        ctk.CTkLabel(
            self.frm_cap, text="Captura de rostro",
            font=("Helvetica", sf(15, self), "bold"), text_color=C_TXT
        ).pack(pady=(s(20, self), s(2, self)))

        self.lbl_cap_nombre = ctk.CTkLabel(
            self.frm_cap, text="",
            font=("Helvetica", sf(12, self)), text_color=C_OK
        )
        self.lbl_cap_nombre.pack(pady=(0, s(8, self)))

        self.lbl_cap_video = ctk.CTkLabel(self.frm_cap, text="", width=s(360, self), height=s(390, self))
        self.lbl_cap_video.pack()

        ctk.CTkLabel(
            self.frm_cap, text="Mueve la cabeza en distintos ángulos",
            font=("Helvetica", sf(11, self)), text_color=C_TXT2
        ).pack(pady=s(6, self))

        self.prog_cap = ctk.CTkProgressBar(
            self.frm_cap, width=s(320, self), height=s(6, self),
            corner_radius=s(3, self), fg_color=C_BORDE, progress_color=C_OK
        )
        self.prog_cap.pack()
        self.prog_cap.set(0)

        self.lbl_cap_cnt = ctk.CTkLabel(
            self.frm_cap, text=f"0 / {FOTOS_CAPTURA} fotos",
            font=("Helvetica", sf(12, self)), text_color=C_TXT2
        )
        self.lbl_cap_cnt.pack(pady=s(4, self))

        self.lbl_cap_instruc = ctk.CTkLabel(
            self.frm_cap, text="",
            font=("Helvetica", sf(11, self)), text_color=C_WARN
        )
        self.lbl_cap_instruc.pack()

        self.lbl_cap_estado = ctk.CTkLabel(
            self.frm_cap, text="",
            font=("Helvetica", sf(12, self), "bold"), text_color=C_OK
        )
        self.lbl_cap_estado.pack(pady=s(4, self))

    def _abrir_captura(self):
        """Abrir interfaz de captura."""
        self._cap_imagenes = []
        self._cap_count = 0
        self._etapa_actual = 0
        self._foto_actual = 0
        self._rostro_detectado_frames = 0
        self._posicion_valida = False
        self._modo = "captura"

        nom = f"{self._reg_datos['nombre']} {self._reg_datos['apellido_p']}"
        self.lbl_cap_nombre.configure(text=nom)
        self.prog_cap.set(0)
        self.lbl_cap_cnt.configure(text=f"0 / {FOTOS_CAPTURA} fotos")
        self.lbl_cap_estado.configure(text=ETAPAS[0]["mensaje"], text_color=C_OK)
        self.lbl_cap_instruc.configure(text="")

        self.frm_form.pack_forget()
        self.frm_cap.pack(fill="both", expand=True)
        self._loop_captura()

    def _detectar(self, gray_small):
        """Detectar rostro en frame redimensionado."""
        if self._detector is None:
            return None
        h_sm, w_sm = gray_small.shape[:2]
        min_size = (
            max(int(w_sm * MIN_TAMANO_RELAT), 20),
            max(int(h_sm * MIN_TAMANO_RELAT), 20)
        )
        rostros = self._detector.detectMultiScale(
            gray_small, scaleFactor=1.1, minNeighbors=MIN_VECINOS, minSize=min_size
        )
        if len(rostros) == 0:
            return None
        return max(rostros, key=lambda r: r[2] * r[3])

    def _loop_captura(self):
        """Loop principal de captura de rostros."""
        if self._modo != "captura" or self._cap_count >= FOTOS_CAPTURA:
            return

        frame = self._camara.leer()
        if frame is None:
            self.after(150, self._loop_captura)
            return

        try:
            h_orig, w_orig = frame.shape[:2]
            base_w = s(360, self)
            base_h = s(390, self)
            target_w = base_w
            target_h = int(base_w * h_orig / w_orig)
            if target_h > base_h:
                target_h = base_h
                target_w = int(base_h * w_orig / h_orig)

            fd = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((base_h, base_w, 3), dtype=np.uint8)
            canvas.fill(10)

            y_off = (base_h - target_h) // 2
            x_off = (base_w - target_w) // 2
            canvas[y_off : y_off + target_h, x_off : x_off + target_w] = fd

            marco_w = int(target_w * 0.60)
            marco_h = int(target_h * 0.70)
            marco_x = x_off + (target_w - marco_w) // 2
            marco_y = y_off + (target_h - marco_h) // 2

            color_marco = (0, 212, 170) if self._posicion_valida else (245, 166, 35)
            cv2.rectangle(canvas, (marco_x, marco_y), (marco_x + marco_w, marco_y + marco_h), color_marco, 3)

            img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            ci = ctk.CTkImage(
                light_image=Image.fromarray(img_rgb),
                dark_image=Image.fromarray(img_rgb),
                size=(base_w, base_h)
            )
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
            ratio = h / h_sm

            if ratio < 0.15:
                instruccion = "Acércate más"
            elif ratio > 0.70:
                instruccion = "Aléjate un poco"
            else:
                cx = (x + w / 2) / w_sm
                cy = (y + h / 2) / h_sm
                if abs(cx - 0.5) > 0.15:
                    instruccion = "Centra tu rostro"
                elif abs(cy - 0.45) > 0.15:
                    instruccion = "Ajusta la altura"
                else:
                    rostro_bien_posicionado = True
                    instruccion = "✓ Posición correcta"

            self._posicion_valida = rostro_bien_posicionado
            if rostro_bien_posicionado:
                self._rostro_detectado_frames += 1
            else:
                self._rostro_detectado_frames = 0

            x1 = max(x + 10, 0)
            y1 = max(y + 10, 0)
            x2 = min(x + w - 10, small.shape[1])
            y2 = min(y + h - 10, small.shape[0])
            rostro_crop = small[y1:y2, x1:x2]

            if rostro_crop.size > 0:
                rostro_res = cv2.resize(rostro_crop, FACE_SIZE)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                rostro_res = clahe.apply(rostro_res)
                rostro_res = cv2.bilateralFilter(rostro_res, 5, 75, 75)
                rostro_res = cv2.normalize(rostro_res, None, 0, 255, cv2.NORM_MINMAX)

                if self._rostro_detectado_frames >= 3:
                    self._cap_imagenes.append(rostro_res)
                    self._foto_actual += 1
                    self._cap_count += 1
                    self._rostro_detectado_frames = 0

                    etapa = ETAPAS[self._etapa_actual]
                    self.prog_cap.set(self._cap_count / FOTOS_CAPTURA)
                    self.lbl_cap_cnt.configure(
                        text=f"{self._cap_count} / {FOTOS_CAPTURA} fotos  |  [{self._foto_actual}/{etapa['fotos']}]",
                        text_color=C_OK
                    )

                    if self._foto_actual >= etapa["fotos"]:
                        self._etapa_actual += 1
                        self._foto_actual = 0
                        if self._etapa_actual >= len(ETAPAS):
                            self.after(150, self._finalizar)
                            return
                        else:
                            self.lbl_cap_instruc.configure(text="Preparando siguiente posición...", text_color=C_WARN)
                            self.lbl_cap_estado.configure(text=ETAPAS[self._etapa_actual]["mensaje"], text_color=C_OK)
                            self.after(2000, self._loop_captura)
                            return
                    self.after(600, self._loop_captura)
                    return
        else:
            self._rostro_detectado_frames = 0
            self._posicion_valida = False
            instruccion = "Acerca tu rostro"

        color = C_OK if self._posicion_valida else C_WARN
        self.lbl_cap_estado.configure(text=ETAPAS[self._etapa_actual]["mensaje"], text_color=color)
        self.lbl_cap_instruc.configure(text=instruccion, text_color=color)
        self.after(150, self._loop_captura)

    def _finalizar(self):
        """Guardar usuario y entrenar modelo."""
        self.lbl_cap_estado.configure(text="Guardando...", text_color=C_WARN)
        self.update()

        try:
            with get_connection() as conn:
                cursor = conn.execute(
                    "INSERT INTO usuarios (nombre, apellido_p, apellido_m, matricula, contrasenia, id_rol) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        self._reg_datos["nombre"],
                        self._reg_datos["apellido_p"],
                        self._reg_datos["apellido_m"],
                        self._reg_datos["matricula"],
                        self._reg_datos["contrasenia"],
                        ROL_ADMIN,
                    ),
                )
                id_u = cursor.lastrowid

            nom_carpeta = f"{self._reg_datos['nombre']}_{self._reg_datos['apellido_p']}"
            carpeta = os.path.join(DATA_DIR, f"{id_u}_{nom_carpeta}")
            os.makedirs(carpeta, exist_ok=True)

            for idx, img in enumerate(self._cap_imagenes):
                cv2.imwrite(os.path.join(carpeta, f"rostro_{idx:03d}.jpg"), img)

            guardar_encoding(id_u, carpeta)

            self.lbl_cap_estado.configure(text="Entrenando modelo...", text_color=C_WARN)
            self.update()
            entrenar()
            self._mostrar_exito()

        except Exception as e:
            print(f"[ERROR SETUP] {e}")
            self.lbl_cap_estado.configure(text=f"Error: {e}", text_color=C_ERROR)

    def _mostrar_exito(self):
        """Mostrar pantalla de éxito."""
        for w in self.frm_cap.winfo_children():
            w.pack_forget()

        inner = ctk.CTkFrame(self.frm_cap, fg_color="transparent")
        inner.pack(expand=True)

        ctk.CTkLabel(inner, text="✅", font=("Helvetica", sf(52, self))).pack(pady=(s(40, self), s(10, self)))
        ctk.CTkLabel(
            inner, text="¡Administrador registrado!",
            font=("Helvetica", sf(20, self), "bold"), text_color=C_OK
        ).pack(pady=(0, s(6, self)))
        ctk.CTkLabel(
            inner, text=f"{self._reg_datos['nombre']} {self._reg_datos['apellido_p']}",
            font=("Helvetica", sf(14, self)), text_color=C_TXT
        ).pack(pady=(0, s(24, self)))

        ctk.CTkButton(
            inner, text="Iniciar sistema  →", width=s(220, self), height=s(46, self),
            fg_color=C_OK, text_color=C_BG, hover_color="#00A88A",
            font=("Helvetica", sf(15, self), "bold"), corner_radius=s(12, self),
            command=self._cerrar
        ).pack()

    def _cerrar(self):
        """Cerrar aplicación."""
        self._modo = "cerrado"
        self._camara.liberar()
        self.destroy()


if __name__ == "__main__":
    app = AdminSetup()
    app.mainloop()
