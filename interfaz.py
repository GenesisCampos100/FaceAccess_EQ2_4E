"""
interfaz.py — Display de acceso para pantalla 7" portrait (480x800px)

Arquitectura correcta:
  · ReconocimientoFacial.py es el motor: maneja cámara, reconocimiento y BD
  · Esta interfaz SOLO muestra lo que el motor produce
  · NO abre su propia cámara — recibe frames ya procesados
  · Comunicación via EstadoSistema (objeto compartido entre hilos)

Para correr:
  python interfaz.py
  (levanta todo: motor de reconocimiento + interfaz juntos)
"""

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
    import numpy as np
    CV2_DISPONIBLE = True
except ImportError:
    CV2_DISPONIBLE = False

# ─── Paleta ───────────────────────────────────────────────────────────────────
C_BG    = "#0F1923"
C_FRAME = "#1A2B3C"
C_FOOT  = "#111E2A"
C_BORDE = "#243447"
C_OK    = "#00D4AA"
C_WARN  = "#F5A623"
C_ERROR = "#E24B4A"
C_TXT   = "#E0EAF4"
C_TXT2  = "#6B8CAE"
C_TXT3  = "#4A6280"

FALLOS_PARA_NUMPAD = 2


class FaceAccessKiosk(ctk.CTk):

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("FaceAccess")
        self.geometry("480x800")
        self.resizable(False, False)
        self.configure(fg_color=C_BG)

        # ── Variables de estado ────────────────────────────────────────────
        self.estado          = "escaneando"
        self.fallos_seguidos = 0
        self.contador_in     = 0
        self.contador_out    = 0
        self._pulso_job      = None
        self._pulso_fase     = 0
        self._prog_job       = None
        self._numpad_val     = ""
        self._numpad_visible = False
        self._ultimo_frame   = None   # frame más reciente del motor
        self.estado_motor    = None   # EstadoSistema de ReconocimientoFacial

        # ── Construir UI ───────────────────────────────────────────────────
        self._ui_header()
        self._ui_saludo()
        self._ui_video()
        self._ui_usuario()

        # ── Arrancar ───────────────────────────────────────────────────────
        self.actualizar_reloj()
        self._pulso_loop()
        self.protocol("WM_DELETE_WINDOW", self._cerrar)
        self._set_estado("escaneando")

        # ── Iniciar motor de reconocimiento ────────────────────────────────
        self.after(300, self._iniciar_motor)

    # ══════════════════════════════════════════════════════════════════════
    #  MOTOR DE RECONOCIMIENTO
    # ══════════════════════════════════════════════════════════════════════

    def _iniciar_motor(self):
        """
        Importa y arranca ReconocimientoFacial en un hilo separado.
        La interfaz no abre cámara — el motor la maneja completamente.
        """
        try:
            from ReconocimientoFacial import EstadoSistema, iniciar_motor_headless

            self.estado_motor = EstadoSistema()

            threading.Thread(
                target=iniciar_motor_headless,
                args=(self.estado_motor,),
                daemon=True
            ).start()

            print("[INFO] Motor de reconocimiento iniciado.")
            self._poll_motor()

        except Exception as e:
            print(f"[ERROR] No se pudo iniciar el motor: {e}")
            import traceback
            traceback.print_exc()

    def _poll_motor(self):
        """
        Consulta cada 30ms el estado del motor y actualiza la UI.
        Recibe el frame ya procesado (con óvalo dibujado) y lo muestra.
        """
        if not self.estado_motor:
            self.after(30, self._poll_motor)
            return

        # ── Mostrar frame de la cámara ─────────────────────────────────────
        frame_proc = self.estado_motor.tomar_frame_procesado()
        if frame_proc is not None and PIL_DISPONIBLE:
            self._ultimo_frame = frame_proc
            try:
                tw = max(self.frame_video.winfo_width(), 2)
                th = max(self.frame_video.winfo_height(), 2)
                h, w = frame_proc.shape[:2]
                esc  = max(tw / w, th / h)
                nw   = max(int(w * esc), 1)
                nh   = max(int(h * esc), 1)
                f    = cv2.resize(frame_proc, (nw, nh),
                                  interpolation=cv2.INTER_AREA)
                xi   = max((nw - tw) // 2, 0)
                yi   = max((nh - th) // 2, 0)
                f    = f[yi:yi+th, xi:xi+tw]
                rgb  = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                img  = Image.fromarray(rgb)
                ci   = ctk.CTkImage(
                    light_image=img, dark_image=img, size=(tw, th)
                )
                self.lbl_video.configure(image=ci, text="")
                self._ci = ci
            except Exception:
                pass

        # ── Leer evento del motor ──────────────────────────────────────────
        if not self._numpad_visible and self.estado not in (
                "exito", "salida", "denegado"):
            evento = self.estado_motor.tomar_evento_ui()
            if evento:
                self._procesar_evento(evento)

        self.after(30, self._poll_motor)

    def _procesar_evento(self, evento: dict):
        """Recibe eventos del motor y actualiza la UI."""
        tipo = evento.get("tipo")

        if tipo == "fase_liveness":
            self._set_badge("● Parpadea", C_WARN)
            self._set_inst("Parpadea para confirmar", C_WARN)
            self._prog_animar(0, 0.4, 1200, C_WARN, "Detección de vida")

        elif tipo == "fase_verificando":
            self._set_badge("● Verificando", C_OK)
            self._set_inst("Verificando identidad...", C_OK)
            self._prog_animar(0.4, 1.0, 600, C_OK, "Verificando identidad")

        elif tipo == "fase_esperando":
            self._set_estado("escaneando")

        elif tipo == "acercate":
            self._set_badge("● Ajusta distancia", C_WARN)
            self._set_inst("Acércate a la cámara", C_WARN)

        elif tipo == "alejate":
            self._set_badge("● Ajusta distancia", C_WARN)
            self._set_inst("Aléjate un poco", C_WARN)

        elif tipo == "acceso_exitoso":
            self.fallos_seguidos = 0
            self.contador_in    += 1
            self._actualizar_contadores()
            self._set_estado(
                "exito",
                nombre   = evento.get("nombre", ""),
                matricula= evento.get("matricula", ""),
                rol      = evento.get("rol", ""),
            )

        elif tipo == "salida_registrada":
            self.fallos_seguidos = 0
            self.contador_out   += 1
            self._actualizar_contadores()
            self._set_estado(
                "salida",
                nombre   = evento.get("nombre", ""),
                matricula= evento.get("matricula", ""),
                rol      = evento.get("rol", ""),
            )

        elif tipo == "acceso_denegado":
            self.fallos_seguidos += 1
            self._set_estado("denegado")
            if self.fallos_seguidos >= FALLOS_PARA_NUMPAD:
                self.after(1500, self._mostrar_numpad)

    # ══════════════════════════════════════════════════════════════════════
    #  CONSTRUCCIÓN UI
    # ══════════════════════════════════════════════════════════════════════

    def _ui_header(self):
        f = ctk.CTkFrame(self, fg_color=C_FRAME, corner_radius=0, height=58)
        f.pack(fill="x")
        f.pack_propagate(False)

        fl = ctk.CTkFrame(f, fg_color="transparent")
        fl.pack(side="left", padx=14, pady=10)

        cv = ctk.CTkCanvas(fl, width=32, height=32,
                            bg=C_FRAME, highlightthickness=0)
        cv.pack(side="left", padx=(0, 8))
        cv.create_oval(2, 2, 30, 30, fill=C_OK, outline="")
        cv.create_text(16, 16, text="FA",
                       fill=C_BG, font=("Helvetica", 10, "bold"))

        fn = ctk.CTkFrame(fl, fg_color="transparent")
        fn.pack(side="left")
        ctk.CTkLabel(fn, text="Mirada ",
                     font=("Helvetica", 13, "bold"),
                     text_color=C_TXT).pack(anchor="w")
        ctk.CTkLabel(fn, text="Control de Acceso",
                     font=("Helvetica", 10),
                     text_color=C_TXT2).pack(anchor="w")

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
        f = ctk.CTkFrame(self, fg_color=C_FOOT, corner_radius=0, height=34)
        f.pack(fill="x")
        f.pack_propagate(False)

        self.lbl_saludo = ctk.CTkLabel(f, text="",
                                        font=("Helvetica", 11),
                                        text_color=C_TXT2)
        self.lbl_saludo.pack(side="left", padx=14)

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

    def _ui_video(self):
        self.frame_video = ctk.CTkFrame(self, fg_color="#080F16",
                                         corner_radius=0, height=532)
        self.frame_video.pack(fill="x")
        self.frame_video.pack_propagate(False)

        # Badge
        self.lbl_badge = ctk.CTkLabel(
            self.frame_video, text="● Escaneando",
            font=("Helvetica", 10, "bold"),
            text_color=C_OK, fg_color=C_FRAME,
            corner_radius=10, padx=10, pady=3
        )
        self.lbl_badge.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)

        # Video
        self.lbl_video = ctk.CTkLabel(
            self.frame_video, text="Iniciando...",
            font=("Helvetica", 13), text_color=C_TXT2
        )
        self.lbl_video.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Instrucción
        self.lbl_inst = ctk.CTkLabel(
            self.frame_video,
            text="Coloca tu rostro en el óvalo",
            font=("Helvetica", 13, "bold"),
            text_color=C_OK, fg_color="#0D1E2D",
            corner_radius=16, padx=16, pady=6
        )
        self.lbl_inst.place(relx=0.5, rely=0.90, anchor="center")

        # Barra de progreso
        self.progress = ctk.CTkProgressBar(
            self.frame_video, width=400, height=4,
            corner_radius=2, fg_color=C_FRAME, progress_color=C_OK
        )
        self.progress.place(relx=0.5, rely=0.97, anchor="center")
        self.progress.set(0)

        self.lbl_prog = ctk.CTkLabel(
            self.frame_video, text="",
            font=("Helvetica", 10), text_color=C_TXT2
        )
        self.lbl_prog.place(relx=0.5, rely=0.93, anchor="center")

        # ── Numpad overlay ─────────────────────────────────────────────────
        self.frame_numpad = ctk.CTkFrame(
            self.frame_video, fg_color="#080F16", corner_radius=0
        )

        ctk.CTkLabel(self.frame_numpad, text="Acceso manual",
                     font=("Helvetica", 15, "bold"),
                     text_color=C_TXT).pack(pady=(28, 2))
        ctk.CTkLabel(self.frame_numpad,
                     text="Reconocimiento no exitoso.\nIngresa tu matrícula.",
                     font=("Helvetica", 11), text_color=C_TXT2,
                     justify="center").pack(pady=(0, 14))

        self.lbl_np_display = ctk.CTkLabel(
            self.frame_numpad, text="",
            font=("Helvetica", 24, "bold"), text_color=C_TXT,
            fg_color=C_FRAME, corner_radius=8,
            width=260, height=48
        )
        self.lbl_np_display.pack(pady=(0, 16))

        fg = ctk.CTkFrame(self.frame_numpad, fg_color="transparent")
        fg.pack()
        for i, txt in enumerate(["1","2","3","4","5","6","7","8","9","⌫","0","OK"]):
            r, c = divmod(i, 3)
            if txt == "OK":
                b = ctk.CTkButton(fg, text=txt, width=76, height=56,
                                   font=("Helvetica", 14, "bold"),
                                   fg_color=C_OK, text_color=C_BG,
                                   hover_color="#00A88A", corner_radius=8,
                                   command=self._np_ok)
            elif txt == "⌫":
                b = ctk.CTkButton(fg, text=txt, width=76, height=56,
                                   font=("Helvetica", 16),
                                   fg_color=C_FRAME, text_color=C_ERROR,
                                   hover_color=C_BORDE,
                                   border_width=1, border_color=C_BORDE,
                                   corner_radius=8,
                                   command=self._np_del)
            else:
                b = ctk.CTkButton(fg, text=txt, width=76, height=56,
                                   font=("Helvetica", 16, "bold"),
                                   fg_color=C_FRAME, text_color=C_TXT,
                                   hover_color=C_BORDE,
                                   border_width=1, border_color=C_BORDE,
                                   corner_radius=8,
                                   command=lambda t=txt: self._np_press(t))
            b.grid(row=r, column=c, padx=5, pady=5)

        ctk.CTkButton(self.frame_numpad, text="Cancelar",
                       font=("Helvetica", 11),
                       fg_color="transparent", text_color=C_TXT2,
                       hover_color=C_FRAME,
                       command=self._np_cancelar).pack(pady=(10, 0))

    def _ui_usuario(self):
        f = ctk.CTkFrame(self, fg_color=C_FRAME,
                          corner_radius=0, height=130)
        f.pack(fill="x", side="bottom")
        f.pack_propagate(False)

        ctk.CTkFrame(f, fg_color=C_BORDE,
                     height=1, corner_radius=0).pack(fill="x")

        cont = ctk.CTkFrame(f, fg_color="transparent")
        cont.pack(fill="both", expand=True, padx=14, pady=10)

        self.lbl_avatar = ctk.CTkLabel(cont, text="",
                                        width=64, height=64)
        self.lbl_avatar.pack(side="left", padx=(0, 12))
        self._avatar_default()

        fd = ctk.CTkFrame(cont, fg_color="transparent")
        fd.pack(side="left", fill="both", expand=True)

        self.lbl_nombre = ctk.CTkLabel(fd, text="Sin identificar",
                                        font=("Helvetica", 15, "bold"),
                                        text_color=C_TXT, anchor="w")
        self.lbl_nombre.pack(anchor="w")

        fp = ctk.CTkFrame(fd, fg_color="transparent")
        fp.pack(anchor="w", pady=3)
        self.pill_mat = ctk.CTkLabel(fp, text="----",
                                      font=("Helvetica", 10),
                                      text_color="#85B7EB",
                                      fg_color="#0C2040",
                                      corner_radius=6, padx=7, pady=2)
        self.pill_mat.pack(side="left", padx=(0, 5))
        self.pill_rol = ctk.CTkLabel(fp, text="---",
                                      font=("Helvetica", 10),
                                      text_color="#AFA9EC",
                                      fg_color="#1C1640",
                                      corner_radius=6, padx=7, pady=2)
        self.pill_rol.pack(side="left")

        self.lbl_sub = ctk.CTkLabel(fd,
                                     text="Esperando reconocimiento...",
                                     font=("Helvetica", 10),
                                     text_color=C_TXT2, anchor="w")
        self.lbl_sub.pack(anchor="w")

        self.lbl_mark = ctk.CTkLabel(cont, text="",
                                      font=("Helvetica", 22),
                                      text_color=C_OK)
        self.lbl_mark.pack(side="right", padx=(0, 4))

    # ══════════════════════════════════════════════════════════════════════
    #  AVATAR
    # ══════════════════════════════════════════════════════════════════════

    def _avatar_default(self):
        if not PIL_DISPONIBLE:
            self.lbl_avatar.configure(text="👤",
                                       font=("Helvetica", 26), image=None)
            return
        size = 64
        img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d    = ImageDraw.Draw(img)
        d.ellipse((0, 0, size-1, size-1), fill="#1A2B3C")
        d.ellipse((22, 8, 42, 28), fill="#4A6280")
        d.ellipse((12, 36, 52, 66), fill="#4A6280")
        ci = ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))
        self.lbl_avatar.configure(image=ci, text="")
        self._av = ci

    def _avatar_usuario(self, ruta=None, color_borde=None):
        if not PIL_DISPONIBLE:
            return
        size        = 64
        color_borde = color_borde or C_OK
        if ruta and os.path.exists(ruta):
            base = Image.open(ruta).convert("RGBA").resize(
                (size, size), Image.Resampling.LANCZOS)
        else:
            r = int(color_borde[1:3], 16) // 2
            g = int(color_borde[3:5], 16) // 2
            b = int(color_borde[5:7], 16) // 2
            base = Image.new("RGBA", (size, size), (r, g, b, 255))
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
        circ = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        circ.paste(base, (0, 0), mask)
        r2 = int(color_borde[1:3], 16)
        g2 = int(color_borde[3:5], 16)
        b2 = int(color_borde[5:7], 16)
        ImageDraw.Draw(circ).ellipse(
            (1, 1, size-2, size-2), outline=(r2, g2, b2, 200), width=3)
        ci = ctk.CTkImage(light_image=circ, dark_image=circ, size=(size, size))
        self.lbl_avatar.configure(image=ci, text="")
        self._av = ci

    # ══════════════════════════════════════════════════════════════════════
    #  NUMPAD
    # ══════════════════════════════════════════════════════════════════════

    def _mostrar_numpad(self):
        if self._numpad_visible:
            return
        self._numpad_val = ""
        self.lbl_np_display.configure(text="")
        self.frame_numpad.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._numpad_visible = True

        # Pausar el motor mientras el numpad está activo
        if self.estado_motor:
            self.estado_motor.en_pausa = True

    def _np_cancelar(self):
        self.frame_numpad.place_forget()
        self._numpad_visible = False
        self.fallos_seguidos = 0
        if self.estado_motor:
            self.estado_motor.en_pausa = False
        self._set_estado("escaneando")

    def _np_press(self, t):
        if len(self._numpad_val) < 12:
            self._numpad_val += t
            self.lbl_np_display.configure(text=self._numpad_val)

    def _np_del(self):
        self._numpad_val = self._numpad_val[:-1]
        self.lbl_np_display.configure(
            text=self._numpad_val if self._numpad_val else "")

    def _np_ok(self):
        if not self._numpad_val:
            return
        matricula = self._numpad_val
        self.frame_numpad.place_forget()
        self._numpad_visible = False
        if self.estado_motor:
            self.estado_motor.en_pausa = False

        from db_manager import (obtener_usuario_por_matricula,
                                 registrar_entrada, guardar_evidencia)
        usuario = obtener_usuario_por_matricula(matricula)
        if usuario:
            from ReconocimientoFacial import guardar_foto_evidencia
            id_acceso = registrar_entrada(usuario["id_usuario"], "MANUAL")
            foto = guardar_foto_evidencia(
                self._ultimo_frame if self._ultimo_frame is not None
                else np.zeros((100, 100, 3), dtype=np.uint8),
                f"manual_{usuario['id_usuario']}"
            )
            guardar_evidencia(id_acceso, foto, "Entrada manual")
            self.fallos_seguidos = 0
            self.contador_in    += 1
            self._actualizar_contadores()
            self._set_estado("exito",
                              nombre   =f"{usuario['nombre']} {usuario['apellido_p']}",
                              matricula=usuario["matricula"],
                              rol      =usuario["nombre_rol"])
        else:
            from db_manager import registrar_intento_fallido
            registrar_intento_fallido(
                f"Acceso manual — matrícula no encontrada: {matricula}",
                matricula=matricula
            )
            self._set_estado("denegado")

    # ══════════════════════════════════════════════════════════════════════
    #  GESTIÓN DE ESTADOS
    # ══════════════════════════════════════════════════════════════════════

    def _set_estado(self, estado, nombre="", matricula="",
                    rol="", ruta_foto=None):
        self.estado = estado
        self._cancel_prog()

        if estado == "escaneando":
            self._set_badge("● Escaneando", C_OK)
            self._set_inst("Coloca tu rostro en el óvalo", C_OK)
            self._prog_set(0, C_OK)
            self.lbl_prog.configure(text="")
            self._reset_usuario()

        elif estado == "exito":
            self._set_badge("✓ Bienvenido/a", C_OK)
            nom = nombre.split()[0] if nombre else "Usuario"
            sal = self._saludo_hora()
            self._set_inst(f"{sal}, {nom}", C_OK)
            self._prog_set(1.0, C_OK)
            self.lbl_prog.configure(text="Acceso registrado")
            self._set_usuario(nombre, matricula, rol,
                               ruta_foto, "Entrada registrada", C_OK, "✓")
            self.after(3500, lambda: self._set_estado("escaneando"))

        elif estado == "salida":
            self._set_badge("◀ Hasta luego", C_WARN)
            nom = nombre.split()[0] if nombre else "Usuario"
            self._set_inst(f"Hasta luego, {nom}", C_WARN)
            self._prog_set(1.0, C_WARN)
            self.lbl_prog.configure(text="Salida registrada")
            self._set_usuario(nombre, matricula, rol,
                               ruta_foto, "Salida registrada", C_WARN, "◀")
            self.after(3500, lambda: self._set_estado("escaneando"))

        elif estado == "denegado":
            self._set_badge("✗ Denegado", C_ERROR)
            self._set_inst("Rostro no registrado", C_ERROR)
            self._prog_set(1.0, C_ERROR)
            self.lbl_nombre.configure(text="No registrado",
                                       text_color=C_ERROR)
            self.pill_mat.configure(text="----")
            self.pill_rol.configure(text="---")
            self.lbl_sub.configure(text="Intento fallido registrado",
                                    text_color=C_ERROR)
            self.lbl_mark.configure(text="✗", text_color=C_ERROR)
            self._avatar_default()
            if self.fallos_seguidos < FALLOS_PARA_NUMPAD:
                self.after(3000, lambda: self._set_estado("escaneando"))

    def _set_badge(self, t, c):
        self.lbl_badge.configure(text=t, text_color=c)

    def _set_inst(self, t, c):
        self.lbl_inst.configure(text=t, text_color=c)

    def _reset_usuario(self):
        self.lbl_nombre.configure(text="Sin identificar",
                                   text_color=C_TXT)
        self.pill_mat.configure(text="----")
        self.pill_rol.configure(text="---")
        self.lbl_sub.configure(text="Esperando reconocimiento...",
                                text_color=C_TXT2)
        self.lbl_mark.configure(text="")
        self._avatar_default()

    def _set_usuario(self, nombre, matricula, rol,
                      ruta, sub, color, marca):
        n = (nombre[:20] + "...") if len(nombre) > 20 else nombre
        self.lbl_nombre.configure(text=n, text_color=C_TXT)
        self.pill_mat.configure(text=matricula or "----")
        self.pill_rol.configure(text=rol or "---")
        self.lbl_sub.configure(text=sub, text_color=color)
        self.lbl_mark.configure(text=marca, text_color=color)
        self._avatar_usuario(ruta, color)

    # ══════════════════════════════════════════════════════════════════════
    #  ANIMACIÓN DE BARRA
    # ══════════════════════════════════════════════════════════════════════

    def _prog_animar(self, desde, hasta, ms, color, label=""):
        self._cancel_prog()
        self.progress.configure(progress_color=color)
        self.lbl_prog.configure(text=label)
        pasos = 30
        iv    = ms // pasos
        inc   = (hasta - desde) / pasos
        self._pv = desde

        def paso():
            self._pv = min(self._pv + inc, hasta)
            self.progress.set(self._pv)
            if self._pv < hasta:
                self._prog_job = self.after(iv, paso)
        self._prog_job = self.after(iv, paso)

    def _prog_set(self, v, color):
        self._cancel_prog()
        self.progress.configure(progress_color=color)
        self.progress.set(v)

    def _cancel_prog(self):
        if self._prog_job:
            self.after_cancel(self._prog_job)
            self._prog_job = None

    # ══════════════════════════════════════════════════════════════════════
    #  PULSO DEL BADGE
    # ══════════════════════════════════════════════════════════════════════

    def _pulso_loop(self):
        if self.estado == "escaneando":
            self._pulso_fase = (self._pulso_fase + 1) % 6
            c = C_OK if self._pulso_fase < 3 else C_TXT3
            self.lbl_badge.configure(text_color=c)
        self._pulso_job = self.after(400, self._pulso_loop)

    # ══════════════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _saludo_hora(self) -> str:
        h = datetime.now().hour
        if h < 12:  return "Buenos días"
        if h < 19:  return "Buenas tardes"
        return "Buenas noches"

    def _actualizar_contadores(self):
        self.lbl_cnt_in.configure(
            text=f"{self.contador_in} entrada"
                 f"{'s' if self.contador_in != 1 else ''}")
        self.lbl_cnt_out.configure(
            text=f"{self.contador_out} salida"
                 f"{'s' if self.contador_out != 1 else ''}")

    def actualizar_reloj(self):
        now = datetime.now()
        self.lbl_hora.configure(text=now.strftime("%H:%M:%S"))
        self.lbl_fecha.configure(text=now.strftime("%d/%m/%Y"))
        h   = now.hour
        sal = ("Buenos días ☀️" if h < 12
               else "Buenas tardes 🌤" if h < 19
               else "Buenas noches 🌙")
        self.lbl_saludo.configure(text=sal)
        self.after(1000, self.actualizar_reloj)

    # ══════════════════════════════════════════════════════════════════════
    #  CIERRE
    # ══════════════════════════════════════════════════════════════════════

    def _cerrar(self):
        if self.estado_motor:
            self.estado_motor.corriendo = False
        if self._pulso_job:
            self.after_cancel(self._pulso_job)
        if self._prog_job:
            self.after_cancel(self._prog_job)
        self.destroy()


if __name__ == "__main__":
    app = FaceAccessKiosk()
    app.mainloop()
