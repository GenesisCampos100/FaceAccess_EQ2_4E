import customtkinter as ctk
from datetime import datetime
import os
import threading

try:
    from PIL import Image, ImageDraw
    PIL_DISPONIBLE = True
except ImportError:
    PIL_DISPONIBLE = False

try:
    import cv2
    CV2_DISPONIBLE = True
except ImportError:
    CV2_DISPONIBLE = False

# ─── Paleta ───────────────────────────────────────────────────────────────────
C_BG        = "#0F1923"
C_FRAME     = "#1A2B3C"
C_FOOTER    = "#111E2A"
C_BORDE     = "#243447"
C_OK        = "#00D4AA"
C_WARN      = "#F5A623"
C_ERROR     = "#E24B4A"
C_TXT       = "#E0EAF4"
C_TXT2      = "#6B8CAE"
C_TXT3      = "#4A6280"

# ─── Configuración ────────────────────────────────────────────────────────────
FALLOS_PARA_NUMPAD = 2   # intentos fallidos antes de mostrar numpad


class FaceAccessKiosk(ctk.CTk):

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")

        self.title("FaceAccess")
        self.geometry("480x800")
        self.resizable(False, False)
        self.configure(fg_color=C_BG)

        # ── Estado del sistema ─────────────────────────────────────────────
        self.estado          = "escaneando"
        self.fallos_seguidos = 0
        self.contador_in     = 0
        self.contador_out    = 0
        self.camara_activa   = False
        self.cap             = None
        self._pulso_job      = None
        self._pulso_fase     = 0
        self._prog_job       = None
        self._prog_val       = 0.0
        self._numpad_val     = ""
        self._numpad_visible = False

        # ── Construcción ───────────────────────────────────────────────────
        self._ui_header()
        self._ui_saludo()
        self._ui_camara()
        self._ui_usuario()

        # ── Inicio ─────────────────────────────────────────────────────────
        self.actualizar_reloj()
        self._pulso_loop()
        self.after(400, self._iniciar_camara)
        self.protocol("WM_DELETE_WINDOW", self._cerrar)
        self._set_estado("escaneando")

    # ══════════════════════════════════════════════════════════════════════
    #  CONSTRUCCIÓN UI
    # ══════════════════════════════════════════════════════════════════════

    def _ui_header(self):
        f = ctk.CTkFrame(self, fg_color=C_FRAME,
                          corner_radius=0, height=58)
        f.pack(fill="x")
        f.pack_propagate(False)

        # Logo
        fl = ctk.CTkFrame(f, fg_color="transparent")
        fl.pack(side="left", padx=14, pady=10)

        canvas = ctk.CTkCanvas(fl, width=32, height=32,
                                bg=C_FRAME, highlightthickness=0)
        canvas.pack(side="left", padx=(0, 8))
        canvas.create_oval(2, 2, 30, 30, fill=C_OK, outline="")
        canvas.create_text(16, 16, text="FA",
                            fill=C_BG, font=("Helvetica", 10, "bold"))

        fn = ctk.CTkFrame(fl, fg_color="transparent")
        fn.pack(side="left")
        ctk.CTkLabel(fn, text="CBTis 163",
                     font=("Helvetica", 13, "bold"),
                     text_color=C_TXT).pack(anchor="w")
        ctk.CTkLabel(fn, text="Control de Acceso",
                     font=("Helvetica", 10),
                     text_color=C_TXT2).pack(anchor="w")

        # Reloj
        fr = ctk.CTkFrame(f, fg_color="transparent")
        fr.pack(side="right", padx=14)
        self.lbl_hora = ctk.CTkLabel(fr, text="",
                                      font=("Helvetica", 18, "bold"),
                                      text_color=C_TXT)
        self.lbl_hora.pack(anchor="e")
        self.lbl_fecha = ctk.CTkLabel(fr, text="",
                                       font=("Helvetica", 10),
                                       text_color=C_TXT2)
        self.lbl_fecha.pack(anchor="e")

    def _ui_saludo(self):
        f = ctk.CTkFrame(self, fg_color=C_FOOTER,
                          corner_radius=0, height=34)
        f.pack(fill="x")
        f.pack_propagate(False)

        # Saludo
        self.lbl_saludo = ctk.CTkLabel(f, text="",
                                        font=("Helvetica", 11),
                                        text_color=C_TXT2)
        self.lbl_saludo.pack(side="left", padx=14)

        # Contadores
        fc = ctk.CTkFrame(f, fg_color="transparent")
        fc.pack(side="right", padx=14)

        ctk.CTkLabel(fc, text="●", font=("Helvetica", 8),
                     text_color=C_OK).pack(side="left", padx=(0, 3))
        self.lbl_cnt_in = ctk.CTkLabel(fc, text="0 entradas",
                                        font=("Helvetica", 10),
                                        text_color=C_OK)
        self.lbl_cnt_in.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(fc, text="●", font=("Helvetica", 8),
                     text_color=C_TXT2).pack(side="left", padx=(0, 3))
        self.lbl_cnt_out = ctk.CTkLabel(fc, text="0 salidas",
                                         font=("Helvetica", 10),
                                         text_color=C_TXT2)
        self.lbl_cnt_out.pack(side="left")

    def _ui_camara(self):
        self.frame_cam = ctk.CTkFrame(self, fg_color="#080F16",
                                       corner_radius=0, height=460)
        self.frame_cam.pack(fill="x")
        self.frame_cam.pack_propagate(False)

        # Badge de estado
        self.lbl_badge = ctk.CTkLabel(
            self.frame_cam, text="● Escaneando",
            font=("Helvetica", 10, "bold"),
            text_color=C_OK, fg_color=C_FRAME,
            corner_radius=10, padx=10, pady=3
        )
        self.lbl_badge.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)

        # Video
        self.lbl_video = ctk.CTkLabel(
            self.frame_cam, text="Iniciando cámara...",
            font=("Helvetica", 13), text_color=C_TXT2
        )
        self.lbl_video.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Instrucción
        self.lbl_inst = ctk.CTkLabel(
            self.frame_cam,
            text="Coloca tu rostro en el óvalo",
            font=("Helvetica", 12, "bold"),
            text_color=C_OK, fg_color="#0D1E2D",
            corner_radius=16, padx=14, pady=5
        )
        self.lbl_inst.place(relx=0.5, rely=0.87, anchor="center")

        # Barra de progreso
        self.progress = ctk.CTkProgressBar(
            self.frame_cam, width=380, height=4,
            corner_radius=2,
            fg_color=C_FRAME, progress_color=C_OK
        )
        self.progress.place(relx=0.5, rely=0.96, anchor="center")
        self.progress.set(0)

        # Label de progreso
        self.lbl_prog = ctk.CTkLabel(
            self.frame_cam, text="",
            font=("Helvetica", 10), text_color=C_TXT2
        )
        self.lbl_prog.place(relx=0.5, rely=0.92, anchor="center")

        # ── Numpad overlay ─────────────────────────────────────────────────
        self.frame_numpad = ctk.CTkFrame(
            self.frame_cam, fg_color="#080F16",
            corner_radius=0
        )
        # No se empaqueta aún — se activa en _mostrar_numpad()

        ctk.CTkLabel(self.frame_numpad,
                     text="Acceso manual",
                     font=("Helvetica", 14, "bold"),
                     text_color=C_TXT).pack(pady=(24, 2))
        ctk.CTkLabel(self.frame_numpad,
                     text="El reconocimiento facial no fue exitoso.\nIngresa tu matrícula.",
                     font=("Helvetica", 11), text_color=C_TXT2,
                     justify="center").pack(pady=(0, 12))

        # Display del numpad
        self.lbl_numpad_display = ctk.CTkLabel(
            self.frame_numpad, text="",
            font=("Helvetica", 22, "bold"),
            text_color=C_TXT,
            fg_color=C_FRAME, corner_radius=8,
            width=240, height=44
        )
        self.lbl_numpad_display.pack(pady=(0, 14))

        # Grid de botones
        frame_grid = ctk.CTkFrame(self.frame_numpad, fg_color="transparent")
        frame_grid.pack()

        teclas = [
            ("1", 0, 0), ("2", 0, 1), ("3", 0, 2),
            ("4", 1, 0), ("5", 1, 1), ("6", 1, 2),
            ("7", 2, 0), ("8", 2, 1), ("9", 2, 2),
            ("⌫", 3, 0), ("0", 3, 1), ("OK", 3, 2),
        ]
        for (txt, row, col) in teclas:
            if txt == "OK":
                btn = ctk.CTkButton(
                    frame_grid, text=txt,
                    width=72, height=52,
                    font=("Helvetica", 14, "bold"),
                    fg_color=C_OK, text_color=C_BG,
                    hover_color="#00A88A",
                    corner_radius=8,
                    command=self._numpad_ok
                )
            elif txt == "⌫":
                btn = ctk.CTkButton(
                    frame_grid, text=txt,
                    width=72, height=52,
                    font=("Helvetica", 16),
                    fg_color=C_FRAME, text_color=C_ERROR,
                    hover_color=C_BORDE,
                    border_width=1, border_color=C_BORDE,
                    corner_radius=8,
                    command=self._numpad_del
                )
            else:
                btn = ctk.CTkButton(
                    frame_grid, text=txt,
                    width=72, height=52,
                    font=("Helvetica", 16, "bold"),
                    fg_color=C_FRAME, text_color=C_TXT,
                    hover_color=C_BORDE,
                    border_width=1, border_color=C_BORDE,
                    corner_radius=8,
                    command=lambda t=txt: self._numpad_press(t)
                )
            btn.grid(row=row, column=col, padx=5, pady=5)

        # Cancelar
        ctk.CTkButton(
            self.frame_numpad,
            text="Cancelar",
            font=("Helvetica", 11),
            fg_color="transparent", text_color=C_TXT2,
            hover_color=C_FRAME,
            command=self._ocultar_numpad
        ).pack(pady=(8, 0))

    def _ui_usuario(self):
        f = ctk.CTkFrame(self, fg_color=C_FRAME,
                          corner_radius=0, height=130)
        f.pack(fill="x", side="bottom")
        f.pack_propagate(False)

        # Separador superior
        ctk.CTkFrame(f, fg_color=C_BORDE,
                     height=1, corner_radius=0).pack(fill="x")

        contenido = ctk.CTkFrame(f, fg_color="transparent")
        contenido.pack(fill="both", expand=True, padx=14, pady=12)

        # Avatar
        self.lbl_avatar = ctk.CTkLabel(
            contenido, text="", width=64, height=64
        )
        self.lbl_avatar.pack(side="left", padx=(0, 12))
        self._avatar_default()

        # Datos
        fd = ctk.CTkFrame(contenido, fg_color="transparent")
        fd.pack(side="left", fill="both", expand=True)

        self.lbl_nombre = ctk.CTkLabel(
            fd, text="Sin identificar",
            font=("Helvetica", 15, "bold"),
            text_color=C_TXT, anchor="w"
        )
        self.lbl_nombre.pack(anchor="w")

        fp = ctk.CTkFrame(fd, fg_color="transparent")
        fp.pack(anchor="w", pady=3)
        self.pill_mat = ctk.CTkLabel(
            fp, text="----",
            font=("Helvetica", 10), text_color="#85B7EB",
            fg_color="#0C2040", corner_radius=6, padx=7, pady=2
        )
        self.pill_mat.pack(side="left", padx=(0, 5))
        self.pill_rol = ctk.CTkLabel(
            fp, text="---",
            font=("Helvetica", 10), text_color="#AFA9EC",
            fg_color="#1C1640", corner_radius=6, padx=7, pady=2
        )
        self.pill_rol.pack(side="left")

        self.lbl_sub = ctk.CTkLabel(
            fd, text="Esperando reconocimiento...",
            font=("Helvetica", 10), text_color=C_TXT2, anchor="w"
        )
        self.lbl_sub.pack(anchor="w")

        # Ícono de resultado
        self.lbl_mark = ctk.CTkLabel(
            contenido, text="",
            font=("Helvetica", 22), text_color=C_OK
        )
        self.lbl_mark.pack(side="right", padx=(0, 4))

    # ══════════════════════════════════════════════════════════════════════
    #  AVATAR
    # ══════════════════════════════════════════════════════════════════════

    def _avatar_default(self):
        if not PIL_DISPONIBLE:
            self.lbl_avatar.configure(text="👤", font=("Helvetica", 26),
                                       image=None)
            return
        size = 64
        img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse((0, 0, size-1, size-1), fill="#1A2B3C")
        draw.ellipse((22, 8, 42, 28), fill="#4A6280")
        draw.ellipse((12, 36, 52, 66), fill="#4A6280")
        ci = ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))
        self.lbl_avatar.configure(image=ci, text="")
        self._av_img = ci

    def _avatar_usuario(self, ruta=None, color_borde=None):
        if not PIL_DISPONIBLE:
            self.lbl_avatar.configure(text="😊", font=("Helvetica", 26),
                                       image=None)
            return
        size        = 64
        color_borde = color_borde or C_OK
        if ruta and os.path.exists(ruta):
            base = Image.open(ruta).convert("RGBA").resize(
                (size, size), Image.Resampling.LANCZOS
            )
        else:
            r = int(color_borde[1:3], 16)
            g = int(color_borde[3:5], 16)
            b = int(color_borde[5:7], 16)
            base = Image.new("RGBA", (size, size), (r//2, g//2, b//2, 255))

        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
        circ = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        circ.paste(base, (0, 0), mask)

        r2 = int(color_borde[1:3], 16)
        g2 = int(color_borde[3:5], 16)
        b2 = int(color_borde[5:7], 16)
        ImageDraw.Draw(circ).ellipse(
            (1, 1, size-2, size-2), outline=(r2, g2, b2, 200), width=3
        )
        ci = ctk.CTkImage(light_image=circ, dark_image=circ, size=(size, size))
        self.lbl_avatar.configure(image=ci, text="")
        self._av_img = ci

    # ══════════════════════════════════════════════════════════════════════
    #  ANIMACIÓN DE PULSO
    # ══════════════════════════════════════════════════════════════════════

    def _pulso_loop(self):
        """
        Simula el pulso del óvalo cambiando el color del badge
        cuando está en modo escaneando.
        """
        if self.estado == "escaneando":
            self._pulso_fase = (self._pulso_fase + 1) % 6
            if self._pulso_fase < 3:
                self.lbl_badge.configure(text_color=C_OK)
            else:
                self.lbl_badge.configure(text_color=C_TXT3)
        self._pulso_job = self.after(400, self._pulso_loop)

    # ══════════════════════════════════════════════════════════════════════
    #  PROGRESO ANIMADO
    # ══════════════════════════════════════════════════════════════════════

    def _prog_animar(self, desde, hasta, ms, color, label=""):
        if self._prog_job:
            self.after_cancel(self._prog_job)
        self.progress.configure(progress_color=color)
        self.lbl_prog.configure(text=label)
        pasos = 30
        iv    = ms // pasos
        inc   = (hasta - desde) / pasos
        self._prog_val = desde

        def paso():
            self._prog_val = min(self._prog_val + inc, hasta)
            self.progress.set(self._prog_val)
            if self._prog_val < hasta:
                self._prog_job = self.after(iv, paso)

        self._prog_job = self.after(iv, paso)

    def _prog_set(self, valor, color):
        if self._prog_job:
            self.after_cancel(self._prog_job)
        self.progress.configure(progress_color=color)
        self.progress.set(valor)

    # ══════════════════════════════════════════════════════════════════════
    #  NUMPAD
    # ══════════════════════════════════════════════════════════════════════

    def _mostrar_numpad(self):
        self._numpad_val = ""
        self.lbl_numpad_display.configure(text="")
        self.frame_numpad.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._numpad_visible = True
        self._set_badge("● Acceso manual", C_WARN)
        self._set_inst("Ingresa tu matrícula", C_WARN)

    def _ocultar_numpad(self):
        self.frame_numpad.place_forget()
        self._numpad_visible = False
        self.fallos_seguidos = 0
        self._set_estado("escaneando")

    def _numpad_press(self, tecla):
        if len(self._numpad_val) < 12:
            self._numpad_val += tecla
            self.lbl_numpad_display.configure(text=self._numpad_val)

    def _numpad_del(self):
        self._numpad_val = self._numpad_val[:-1]
        self.lbl_numpad_display.configure(
            text=self._numpad_val if self._numpad_val else ""
        )

    def _numpad_ok(self):
        if not self._numpad_val:
            return
        matricula = self._numpad_val
        self._ocultar_numpad()

        # ── Conectar con db_manager ──────────────────────────────────────
        # from db_manager import obtener_usuario_por_matricula, registrar_entrada, guardar_evidencia
        # usuario = obtener_usuario_por_matricula(matricula)
        # if usuario:
        #     id_acceso = registrar_entrada(usuario['id_usuario'], 'MANUAL')
        #     self._set_estado('exito',
        #         nombre=f"{usuario['nombre']} {usuario['apellido_p']}",
        #         matricula=usuario['matricula'],
        #         rol=usuario['nombre_rol'])
        # else:
        #     self._set_estado('denegado')
        print(f"[MANUAL] Matrícula ingresada: {matricula}")

    # ══════════════════════════════════════════════════════════════════════
    #  GESTIÓN DE ESTADOS — PUNTO DE INTEGRACIÓN PRINCIPAL
    # ══════════════════════════════════════════════════════════════════════

    def _set_estado(self, estado, nombre="", matricula="",
                    rol="", ruta_foto=None, ultimo_acceso=""):
        """
        Llama este método desde el hilo de reconocimiento.

        Estados: 'escaneando' | 'liveness' | 'verificando' |
                 'exito' | 'salida' | 'denegado'
        """
        self.estado = estado

        if estado == "escaneando":
            self._set_badge("● Escaneando", C_OK)
            self._set_inst("Coloca tu rostro en el óvalo", C_OK)
            self._prog_set(0, C_OK)
            self.lbl_prog.configure(text="")
            self._reset_usuario()

        elif estado == "liveness":
            self._set_badge("● Parpadea", C_WARN)
            self._set_inst("Parpadea para confirmar", C_WARN)
            self._prog_animar(0, 0.4, 1200, C_WARN, "Detección de vida")

        elif estado == "verificando":
            self._set_badge("● Verificando", C_OK)
            self._set_inst("Verificando identidad...", C_OK)
            self._prog_animar(0.4, 1.0, 600, C_OK, "Verificando identidad")

        elif estado == "exito":
            self.fallos_seguidos = 0
            self.contador_in    += 1
            self._actualizar_contadores()
            self._set_badge("✓ Bienvenido/a", C_OK)
            nom_corto = nombre.split()[0] if nombre else "Usuario"
            saludo    = self._saludo_hora()
            self._set_inst(f"{saludo}, {nom_corto}", C_OK)
            self._prog_set(1.0, C_OK)
            self.lbl_prog.configure(text="Acceso registrado")
            self._set_usuario(nombre, matricula, rol, ruta_foto,
                               ultimo_acceso or "Entrada registrada",
                               C_OK, "✓")
            self.after(3500, lambda: self._set_estado("escaneando"))

        elif estado == "salida":
            self.fallos_seguidos = 0
            self.contador_out   += 1
            self._actualizar_contadores()
            self._set_badge("◀ Hasta luego", C_WARN)
            nom_corto = nombre.split()[0] if nombre else "Usuario"
            self._set_inst(f"Hasta luego, {nom_corto}", C_WARN)
            self._prog_set(1.0, C_WARN)
            self.lbl_prog.configure(text="Salida registrada")
            self._set_usuario(nombre, matricula, rol, ruta_foto,
                               ultimo_acceso or "Salida registrada",
                               C_WARN, "◀")
            self.after(3500, lambda: self._set_estado("escaneando"))

        elif estado == "denegado":
            self.fallos_seguidos += 1
            self._set_badge("✗ Denegado", C_ERROR)
            self._set_inst("Rostro no registrado", C_ERROR)
            self._prog_set(1.0, C_ERROR)
            self.lbl_prog.configure(text="Intento fallido")
            self.lbl_nombre.configure(text="No registrado",
                                       text_color=C_ERROR)
            self.pill_mat.configure(text="----")
            self.pill_rol.configure(text="---")
            self.lbl_sub.configure(text="Intento fallido registrado",
                                    text_color=C_ERROR)
            self.lbl_mark.configure(text="✗", text_color=C_ERROR)
            self._avatar_default()

            # Mostrar numpad si supera el límite de fallos
            if self.fallos_seguidos >= FALLOS_PARA_NUMPAD:
                self.after(1500, self._mostrar_numpad)
            else:
                self.after(3000, lambda: self._set_estado("escaneando"))

    def _set_badge(self, texto, color):
        self.lbl_badge.configure(text=texto, text_color=color)

    def _set_inst(self, texto, color):
        self.lbl_inst.configure(text=texto, text_color=color)

    def _reset_usuario(self):
        self.lbl_nombre.configure(text="Sin identificar",
                                   text_color=C_TXT)
        self.pill_mat.configure(text="----")
        self.pill_rol.configure(text="---")
        self.lbl_sub.configure(text="Esperando reconocimiento...",
                                text_color=C_TXT2)
        self.lbl_mark.configure(text="")
        self._avatar_default()

    def _set_usuario(self, nombre, matricula, rol, ruta_foto,
                      sub_texto, color, marca):
        n = (nombre[:20] + "...") if len(nombre) > 20 else nombre
        self.lbl_nombre.configure(text=n, text_color=C_TXT)
        self.pill_mat.configure(text=matricula or "----")
        self.pill_rol.configure(text=rol or "---")
        self.lbl_sub.configure(text=sub_texto, text_color=color)
        self.lbl_mark.configure(text=marca, text_color=color)
        self._avatar_usuario(ruta_foto, color)

    # ══════════════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _saludo_hora(self) -> str:
        h = datetime.now().hour
        if h < 12:
            return "Buenos días"
        elif h < 19:
            return "Buenas tardes"
        return "Buenas noches"

    def _actualizar_contadores(self):
        self.lbl_cnt_in.configure(
            text=f"{self.contador_in} entrada{'s' if self.contador_in != 1 else ''}"
        )
        self.lbl_cnt_out.configure(
            text=f"{self.contador_out} salida{'s' if self.contador_out != 1 else ''}"
        )

    def actualizar_reloj(self):
        now = datetime.now()
        self.lbl_hora.configure(text=now.strftime("%H:%M:%S"))
        self.lbl_fecha.configure(text=now.strftime("%d/%m/%Y"))

        # Actualizar saludo
        h = now.hour
        if h < 12:
            sal = "Buenos días ☀️"
        elif h < 19:
            sal = "Buenas tardes 🌤"
        else:
            sal = "Buenas noches 🌙"
        self.lbl_saludo.configure(text=sal)

        self.after(1000, self.actualizar_reloj)

    # ══════════════════════════════════════════════════════════════════════
    #  CÁMARA
    # ══════════════════════════════════════════════════════════════════════

    def _iniciar_camara(self):
        if not CV2_DISPONIBLE or not PIL_DISPONIBLE:
            self.lbl_video.configure(text="Dependencias no instaladas")
            return

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.lbl_video.configure(text="No se pudo abrir la cámara")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camara_activa = True
        self._loop_camara()

    def _loop_camara(self):
        if not self.camara_activa or not self.cap:
            return
        if self._numpad_visible:
            self.after(50, self._loop_camara)
            return

        ok, frame = self.cap.read()
        if not ok:
            self.after(50, self._loop_camara)
            return

        frame = cv2.flip(frame, 1)
        tw    = max(self.frame_cam.winfo_width(), 2)
        th    = max(self.frame_cam.winfo_height(), 2)
        h, w  = frame.shape[:2]
        esc   = max(tw / w, th / h)
        nw, nh = max(int(w * esc), 1), max(int(h * esc), 1)
        frame  = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        xi     = max((nw - tw) // 2, 0)
        yi     = max((nh - th) // 2, 0)
        frame  = frame[yi:yi+th, xi:xi+tw]
        frame  = self._dibujar_ovalo(frame)

        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(rgb)
        ci    = ctk.CTkImage(light_image=img, dark_image=img, size=(tw, th))
        self.lbl_video.configure(image=ci, text="")
        self._ci = ci

        self.after(50, self._loop_camara)

    def _dibujar_ovalo(self, frame):
        """Óvalo guía con color según estado y efecto de pulso."""
        h, w  = frame.shape[:2]
        cx, cy = w // 2, h // 2
        rx, ry = int(w * 0.28), int(h * 0.44)

        colores = {
            "escaneando" : (160, 160, 160),
            "liveness"   : (35,  166, 245),
            "verificando": (0,   212, 170),
            "exito"      : (0,   212, 170),
            "salida"     : (35,  166, 245),
            "denegado"   : (74,  75,  226),
        }
        color = colores.get(self.estado, (120, 120, 120))

        # Pulso externo en modo escaneando
        if self.estado == "escaneando":
            alpha = 0.3 + 0.2 * (self._pulso_fase / 5)
            ov    = frame.copy()
            cv2.ellipse(ov, (cx, cy),
                        (rx + 10, ry + 10), 0, 0, 360, color, 1)
            cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

        # Óvalo principal
        cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, color, 2)

        return frame

    # ══════════════════════════════════════════════════════════════════════
    #  CIERRE
    # ══════════════════════════════════════════════════════════════════════

    def _cerrar(self):
        self.camara_activa = False
        if self._pulso_job:
            self.after_cancel(self._pulso_job)
        if self._prog_job:
            self.after_cancel(self._prog_job)
        if self.cap:
            self.cap.release()
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRACIÓN CON ReconocimientoFacial.py
# ══════════════════════════════════════════════════════════════════════════════
"""
Agrega esto en __init__ después de self._set_estado("escaneando"):

    from entrenadoRF import cargar_encodings_bd
    from ReconocimientoFacial import hilo_reconocimiento, EstadoSistema

    self.enc_bd, self.ids_bd = cargar_encodings_bd()
    self.estado_rf = EstadoSistema()

    threading.Thread(
        target=hilo_reconocimiento,
        args=(self.estado_rf, self.enc_bd, self.ids_bd),
        daemon=True
    ).start()
    self._poll_reconocimiento()

Agrega este método a la clase:

    def _poll_reconocimiento(self):
        res = self.estado_rf.tomar_resultado_hilo()
        if res and res.get('tipo') == 'rostro':
            from db_manager import obtener_usuario_por_id, tiene_entrada_abierta
            id_u = res.get('id_usuario')
            if id_u:
                u      = obtener_usuario_por_id(id_u)
                nombre = f"{u['nombre']} {u['apellido_p']}"
                estado = 'exito' if not tiene_entrada_abierta(id_u) else 'salida'
                self._set_estado(estado, nombre=nombre,
                                 matricula=u['matricula'],
                                 rol=u['nombre_rol'])
            else:
                self._set_estado('denegado')
        self.after(100, self._poll_reconocimiento)
"""

if __name__ == "__main__":
    app = FaceAccessKiosk()

    # Simulación de flujo completo para pruebas de diseño
    app.after(2500, lambda: app._set_estado("liveness"))
    app.after(4500, lambda: app._set_estado("verificando"))
    app.after(6000, lambda: app._set_estado(
        "exito",
        nombre="Genesis Martinez",
        matricula="ALU2024001",
        rol="Alumno",
        ultimo_acceso="Hoy 08:15 AM"
    ))

    app.mainloop()
