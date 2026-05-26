"""
ReconocimientoFacial.py
Clase principal FaceAccess — orquesta los módulos, no implementa lógica propia.

Módulos externos:
    core/reconocimiento.py  → buscar(), cargar_encodings()
    core/camara.py          → CamaraManager
    ui/constantes.py        → colores, dimensiones, parámetros
    ui/teclado.py           → TecladoVirtual
    database/db_manager.py  → todas las operaciones de BD
    models/entrenadoRF.py   → entrenar(), cargar_modelo_lbph()
"""

import cv2
import os
import threading
import time
import numpy as np
from queue import Queue
import customtkinter as ctk
from PIL import Image
from collections import Counter
from datetime import datetime
from ui.panel_admin import PanelAdmin

# ── Módulos propios ───────────────────────────────────────────────────────────
from core.reconocimiento import cargar_encodings, buscar
from core.camara         import CamaraManager
from ui.constantes       import (
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

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("FaceAccess")
        self.geometry(APP_GEOMETRY)
        self.resizable(False, False)
        self.configure(fg_color=C_BG)

        # ── Estado de la aplicación ───────────────────────────────────────────
        self.estado          = "escaneando"
        self.fallos          = 0
        self._esperando_numpad = False
        self.cnt_in      = 0
        self._pulso_fase = 0
        self._pulso_job  = None
        self._ci         = None
        self._np_val     = ""
        self._np_vis     = False

        # ── Locks y queue thread-safe ─────────────────────────────────────────
        self._lock           = threading.Lock()
        self._lock_resultado = threading.Lock()
        self._lock_coords    = threading.Lock()
        self._queue_frames   = Queue(maxsize=2)

        # ── Variables internas de flujo ───────────────────────────────────────
        self._resultado     = None
        self._res_nuevo     = False
        self._ultimo_frame  = None
        self._en_pausa      = False
        self._t_pausa       = None
        self._buffer        = []
        self._frames_desc   = 0
        self._ultimo_coords = None
        self._ultimo_id_u   = None
        self._t_ultimo_res  = 0.0
        self._buffer_varianza = []  # Anti-spoofing: detecta imágenes estáticas

        # ── Modelo LBPH ───────────────────────────────────────────────────────
        self._recognizer, _ = cargar_encodings()

        # ── Modos y flujos ────────────────────────────────────────────────────
        self._modo                     = "acceso"
        self._login_usuario            = None
        self._login_validando          = False
        self._login_frames_confirmados = 0
        self._login_frames_fallidos    = 0
        self._cap_imagenes             = []
        self._cap_count                = 0
        self._reg_datos                = {}
        self._coincidencias            = []
        self._etapas_captura           = []
        self._etapa_actual             = 0
        self._foto_actual              = 0
        self._rostro_detectado_frames  = 0
        self._posicion_valida          = False
        self._logo_clicks              = 0
        self._logo_timer               = None

        # ── Teclado táctil virtual ────────────────────────────────────────────
        self._teclado = TecladoVirtual(root=self)

        # ── Dimensiones internas de numpad (compatibilidad) ───────────────────
        self._KB_BW    = (KB_APP_W - KB_PAD * (KB_COLS + 1)) // KB_COLS
        self._KB_BH    = KB_BH
        self._KB_FS    = KB_FS
        self._KB_PAD   = KB_PAD
        self._KB_ACT_H = KB_ACT_H
        self._KB_SPC_W = KB_APP_W - 160

        # ── Cámara ────────────────────────────────────────────────────────────
        self._camara = CamaraManager()

        # ── Construir UI ──────────────────────────────────────────────────────
        self._build_header()
        self._build_saludo()
        self._build_video()
        self._teclado = TecladoVirtual(root=self.frame_video)
        self._build_overlays()

        self._detector = self._init_haar()
        self._update_clock()
        self._pulso()
        self.protocol("WM_DELETE_WINDOW", self._cerrar)
        self.after(400, self._iniciar)

    # ── Inicialización ────────────────────────────────────────────────────────

    def _init_haar(self):
        detector = cv2.CascadeClassifier(HAAR_CASCADE)
        if detector.empty():
            print(f"[ERROR] No se encontró Haar Cascade: {HAAR_CASCADE}")
            return None
        print("[MOTOR] Haar Cascade activo.")
        return detector

    def _iniciar(self):
        self._camara.iniciar(lock=self._lock)
        threading.Thread(target=self._hilo_rec, daemon=True).start()
        self._loop_camara()
        from database.db_manager import get_connection, ROL_ADMIN
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as total FROM usuarios WHERE id_rol=? AND estatus=1",
                (ROL_ADMIN,)).fetchone()
        self._admin_existe = row["total"] > 0
        if self._admin_existe:
            self._loop_logica()

    # ── Setup inicial (primer admin) ──────────────────────────────────────────

    def _setup_inicial(self):
        """Se llama desde main.py solo cuando la BD está vacía. Muestra formulario admin."""
        import shutil
        # Limpiar modelo y rostros viejos para evitar falso duplicado
        for xml in [
            os.path.join("database", "modelo_lbph.xml"),
            os.path.join("models",   "modelo_lbph.xml"),
            "modelo_lbph.xml",
        ]:
            if os.path.exists(xml):
                os.remove(xml)
        data_dir = os.path.join("database", "data_rostros")
        if os.path.isdir(data_dir):
            shutil.rmtree(data_dir)
            os.makedirs(data_dir, exist_ok=True)
        self._recognizer = None  # Sin modelo = sin validación de duplicado

        self._modo = "registro"
        self._en_pausa = True
        self.combo_rol.configure(values=["ADMIN"])
        self.combo_rol.set("ADMIN")
        self._actualizar_campos_rol("ADMIN")
        if self._grado_frame: self._grado_frame.grid_remove()
        if self._grupo_frame: self._grupo_frame.grid_remove()
        self.lbl_reg_op.configure(
            text="⚙ Setup inicial — Crear primer administrador", text_color=C_WARN)
        self.lbl_reg_err.configure(text="")
        for e in self._entries.values():
            e.delete(0, "end")
        self._ocultar_overlays()
        self.ov_registro.place(relx=0, rely=0, relwidth=1, relheight=1)

    def _reg_continuar(self):
        """Override de continuar: si es setup inicial usa ROL_ADMIN, si no flujo normal."""
        datos = {k: e.get().strip() for k, e in self._entries.items()}
        mapa  = {"ADMIN":1, "PERSONAL_AUTORIZADO":2, "PERSONAL_ESCOLAR":3, "ALUMNO":4}
        rol_sel = self.combo_rol.get()

        if not all([datos["nombre"], datos["apellido_p"],
                    datos["matricula"], datos["contrasenia"]]):
            self.lbl_reg_err.configure(
                text="Nombre, apellido, matrícula y contraseña son obligatorios."); return

        if rol_sel == "ALUMNO":
            if not datos.get("grado") or not datos.get("grupo"):
                self.lbl_reg_err.configure(
                    text="Grado y grupo son obligatorios para alumnos."); return
            if not datos["grado"].isdigit():
                self.lbl_reg_err.configure(
                    text="El grado solo debe ser un número (ej: 1, 2, 10)."); return
            if not datos["grupo"].isalpha() or len(datos["grupo"]) != 1:
                self.lbl_reg_err.configure(
                    text="El grupo solo debe ser una letra (A, B, C...)."); return
            if not (1 <= int(datos["grado"]) <= 12):
                self.lbl_reg_err.configure(
                    text="El grado debe estar entre 1 y 12."); return

        if not datos["nombre"].replace(" ", "").isalpha():
            self.lbl_reg_err.configure(
                text="El nombre solo debe contener letras."); return
        if not datos["apellido_p"].replace(" ", "").isalpha():
            self.lbl_reg_err.configure(
                text="El apellido paterno solo debe contener letras."); return
        if datos["apellido_m"] and not datos["apellido_m"].replace(" ", "").isalpha():
            self.lbl_reg_err.configure(
                text="El apellido materno solo debe contener letras."); return
        if obtener_usuario_por_matricula(datos["matricula"].upper()):
            self.lbl_reg_err.configure(
                text=f"La matrícula '{datos['matricula'].upper()}' ya existe."); return
        if len(datos["contrasenia"]) < 6:
            self.lbl_reg_err.configure(
                text="La contraseña debe tener mínimo 6 caracteres."); return
        if datos["contrasenia"] != datos.get("contrasenia2", ""):
            self.lbl_reg_err.configure(
                text="Las contraseñas no coinciden."); return

        datos["nombre"]     = datos["nombre"].upper()
        datos["apellido_p"] = datos["apellido_p"].upper()
        datos["apellido_m"] = datos["apellido_m"].upper() if datos["apellido_m"] else ""
        datos["matricula"]  = datos["matricula"].upper()
        datos["grado"]      = datos.get("grado", "").upper()
        datos["grupo"]      = datos.get("grupo", "").upper()
        datos["id_rol"]     = mapa.get(rol_sel, 4)
        self._reg_datos = datos
        self._abrir_captura()

    # ── Hilo de reconocimiento ────────────────────────────────────────────────

    def _hilo_rec(self):
        """
        Hilo dedicado. Lee frames de la Queue, detecta con Haar y reconoce con LBPH.
        Llama a buscar() de core/reconocimiento.py con el crop CRUDO.
        """
        while True:
            try:
                frame = self._queue_frames.get(timeout=0.5)
            except:
                
                continue
            

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (0, 0), fx=ESCALA_DETEC, fy=ESCALA_DETEC)

            coords = self._detectar(small, frame)

            def sin_rostro():
                with self._lock_resultado:
                    self._resultado    = {"tipo": "sin_rostro"}
                    self._res_nuevo    = True
                    self._t_ultimo_res = time.monotonic()

            if coords is None:
                sin_rostro(); continue

            x_s, y_s, w_s, h_s = coords
            h_sm, w_sm = small.shape[:2]

            if h_s / h_sm < MIN_TAMANO_RELAT or w_s / w_sm < MIN_TAMANO_RELAT:
                sin_rostro(); continue

            rostro_gray = small[y_s:y_s+h_s, x_s:x_s+w_s]
            if rostro_gray.size > 0:

                # ── Anti-spoofing: detectar imagen estática ───────────────────
                varianza = cv2.Laplacian(rostro_gray, cv2.CV_64F).var()
                self._buffer_varianza.append(varianza)
                if len(self._buffer_varianza) > 12:
                    self._buffer_varianza.pop(0)
                if len(self._buffer_varianza) >= 12:
                    variacion = float(np.std(self._buffer_varianza))
                    if variacion < 1.5:  # Imagen estática: varianza constante
                        with self._lock_resultado:
                            self._resultado    = {"tipo": "imagen_estatica"}
                            self._res_nuevo    = True
                            self._t_ultimo_res = time.monotonic()
                        continue
                # ─────────────────────────────────────────────────────────────
                id_u, confianza = buscar(rostro_gray, self._recognizer)
            else:
                id_u, confianza = None, 999

            esc = 1.0 / ESCALA_DETEC
            coords_orig = (
                int(y_s * esc),
                int((x_s + w_s) * esc),
                int((y_s + h_s) * esc),
                int(x_s * esc),
            )

            with self._lock_resultado:
                self._resultado = {
                    "tipo"      : "rostro",
                    "id_usuario": id_u,
                    "confianza" : confianza,
                    "frame_cap" : frame.copy(),
                    "coords"    : coords_orig,
                }
                self._res_nuevo    = True
                self._t_ultimo_res = time.monotonic()

    def _detectar(self, gray_small, frame_original=None):
        if self._detector is None:
            return None

        h_sm, w_sm = gray_small.shape[:2]
        min_size = (
            max(int(w_sm * MIN_TAMANO_RELAT), 20),
            max(int(h_sm * MIN_TAMANO_RELAT), 20)
        )

        with self._lock_coords:
            last = self._ultimo_coords

        if last is not None:
            top_o, right_o, bottom_o, left_o = last
            pad = 0.4
            x_roi  = max(int(left_o   * ESCALA_DETEC * (1 - pad)), 0)
            y_roi  = max(int(top_o    * ESCALA_DETEC * (1 - pad)), 0)
            x2_roi = min(int(right_o  * ESCALA_DETEC * (1 + pad)), w_sm)
            y2_roi = min(int(bottom_o * ESCALA_DETEC * (1 + pad)), h_sm)
            roi = gray_small[y_roi:y2_roi, x_roi:x2_roi]
            if roi.size > 0:
                rostros_roi = self._detector.detectMultiScale(
                    roi, scaleFactor=1.1, minNeighbors=MIN_VECINOS, minSize=min_size)
                if len(rostros_roi) > 0:
                    x, y, w, h = max(rostros_roi, key=lambda r: r[2] * r[3])
                    return (x + x_roi, y + y_roi, w, h)

        rostros = self._detector.detectMultiScale(
            gray_small, scaleFactor=1.1, minNeighbors=MIN_VECINOS, minSize=min_size)

        if len(rostros) == 0:
            return None
        return max(rostros, key=lambda r: r[2] * r[3])

    # ── Loop cámara ───────────────────────────────────────────────────────────

    def _loop_camara(self):
        if self._modo == "captura":
            self.after(100, self._loop_camara)
            return
        if self._np_vis or self._modo not in ("acceso", "login"):
            self.after(33, self._loop_camara)
            return

        frame = self._camara.leer()
        if frame is None:
            self.after(33, self._loop_camara)
            return

        with self._lock:
            self._ultimo_frame = frame.copy()

        try:
            tw = max(self.frame_video.winfo_width(), 2)
            th = max(self.frame_video.winfo_height(), 2)
            h, w = frame.shape[:2]
            esc  = max(tw/w, th/h)
            nw, nh = max(int(w*esc), 1), max(int(h*esc), 1)
            f    = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            xi   = max((nw-tw)//2, 0)
            yi   = max((nh-th)//2, 0)
            f    = f[yi:yi+th, xi:xi+tw]

            if not self._en_pausa and self._modo == "acceso":
                try:
                    self._queue_frames.put_nowait(f.copy())
                    print("[CAMARA] Frame enviado a queue")  # ← esto
                except:
                    pass

            if self._modo == "acceso":
                f = self._dibujar_rect(f, tw, th, nw, nh, xi, yi)
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            ci  = ctk.CTkImage(light_image=img, dark_image=img, size=(tw, th))
            self.lbl_video.configure(image=ci, text="")
            self._ci = ci
        except Exception as e:
            print(f"[VIDEO ERROR] {e}")

        self.after(33, self._loop_camara)

    def _dibujar_rect(self, frame, tw, th, nw, nh, xi, yi):
        with self._lock_coords:
            coords = self._ultimo_coords

        if coords is None or self._en_pausa:
            return frame

        top_s, r_s, b_s, l_s = coords
        top_s = max(top_s, 0); b_s = min(b_s, th)
        l_s   = max(l_s,   0); r_s = min(r_s, tw)

        if b_s <= top_s or r_s <= l_s:
            return frame

        colores = {
            "escaneando" : (180, 180, 180),
            "verificando": (0,   212, 170),
            "exito"      : (0,   212, 170),
            "salida"     : (35,  166, 245),
            "denegado"   : (74,  75,  226),
        }
        color  = colores.get(self.estado, (180, 180, 180))
        largo  = max(min(int((r_s - l_s) * 0.20), 30), 15)
        grosor = 3

        cv2.line(frame, (l_s, top_s), (l_s + largo, top_s), color, grosor)
        cv2.line(frame, (l_s, top_s), (l_s, top_s + largo), color, grosor)
        cv2.line(frame, (r_s, top_s), (r_s - largo, top_s), color, grosor)
        cv2.line(frame, (r_s, top_s), (r_s, top_s + largo), color, grosor)
        cv2.line(frame, (l_s, b_s),   (l_s + largo, b_s),   color, grosor)
        cv2.line(frame, (l_s, b_s),   (l_s, b_s - largo),   color, grosor)
        cv2.line(frame, (r_s, b_s),   (r_s - largo, b_s),   color, grosor)
        cv2.line(frame, (r_s, b_s),   (r_s, b_s - largo),   color, grosor)

        if self.estado == "escaneando":
            alpha = 0.15 + 0.12 * (self._pulso_fase / 5)
            ov    = frame.copy()
            cv2.rectangle(ov, (l_s, top_s), (r_s, b_s), color, 1)
            cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

        return frame

    # ── Loop lógica ───────────────────────────────────────────────────────────

    def _loop_logica(self):
        if self._modo != "acceso":
            self.after(100, self._loop_logica)
            return

        if self._en_pausa:
            if (datetime.now() - self._t_pausa).total_seconds() >= PAUSA_SEG:
                self._en_pausa    = False
                self._buffer      = []
                self._frames_desc = 0
                self._set_estado("escaneando")
                self._ocultar_msg()
            self.after(100, self._loop_logica)
            return

        with self._lock_resultado:
            if not self._res_nuevo:
                edad = time.monotonic() - self._t_ultimo_res
                if edad > 0.4 and self._t_ultimo_res > 0:
                    self._buffer      = []
                    self._frames_desc = 0
                    with self._lock_coords:
                        self._ultimo_coords = None
                        self._ultimo_id_u   = None
                    if self.estado != "escaneando":
                        self.after(0, lambda: (self._set_estado("escaneando"),
                                               self._ocultar_msg()))
                self.after(100, self._loop_logica)
                return
            res = self._resultado
            self._res_nuevo = False

        if res["tipo"] == "sin_rostro":
            self._buffer      = []
            self._frames_desc = 0
            with self._lock_coords:
                self._ultimo_coords = None
                self._ultimo_id_u   = None
            if self.estado != "escaneando":
                self._set_estado("escaneando")
                self._ocultar_msg()
            self.after(100, self._loop_logica)
            return

        if res["tipo"] == "imagen_estatica":
            self._buffer      = []
            self._frames_desc = 0
            self._set_badge("⚠ Imagen detectada", C_ERROR)
            self.after(100, self._loop_logica)
            return

        id_u      = res["id_usuario"]
        confianza = res.get("confianza", 999.0)

        with self._lock_coords:
            self._ultimo_coords = res.get("coords")
            self._ultimo_id_u   = id_u

        if id_u is not None:
            self._frames_desc = 0
            self._buffer.append((id_u, confianza))
            frames_necesarios = 1 if confianza < 45 else FRAMES_CONFIRM
            if len(self._buffer) < frames_necesarios:
                self._set_badge("● Verificando", C_OK)
                self.after(100, self._loop_logica)
                return
            ids_buffer = [item[0] for item in self._buffer]
            conteo     = Counter(ids_buffer)
            id_final   = conteo.most_common(1)[0][0]
            self._buffer = []
            self._registrar_acceso(id_final, res["frame_cap"])
        else:
            self._buffer = []
            self._frames_desc += 1
            if self._frames_desc >= FRAMES_CONFIRM:
                self._frames_desc = 0
                self._registrar_acceso(None, res["frame_cap"])
            else:
                self._set_badge("● Detectando", C_WARN)

        self.after(100, self._loop_logica)

    def _registrar_acceso(self, id_usuario, frame_cap):
        dt = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")

        if id_usuario is not None:
            u      = obtener_usuario_por_id(id_usuario)
            nombre = f"{u['nombre']} {u['apellido_p']}" if u else f"ID {id_usuario}"
            mat    = u["matricula"] if u else ""
            if u and u["id_rol"] == ROL_ALUMNO:
                grado = (u["grado"] or "").strip()
                grupo = (u["grupo"] or "").strip()
                rol   = f"{grado}  {grupo}".strip() if (grado or grupo) else u["nombre_rol"]
            else:
                rol = u["nombre_rol"] if u else ""

            registrar_entrada(id_usuario, "FACIAL")
            print(f"[BD] ACCESO FACIAL — {nombre} | {dt}")
            self.cnt_in += 1; self._upd_cnt()
            self._set_estado("exito")
            self._mostrar_msg("exito", nombre, mat, rol)
        else:
            registrar_intento_fallido("Rostro no reconocido")
            print(f"[BD] DENEGADO | {dt}")
            self.fallos += 1
            self._set_estado("denegado")
            self._mostrar_msg("denegado")
            if self.fallos >= FALLOS_NUMPAD and not self._esperando_numpad:
                self._esperando_numpad = True
                self._en_pausa = True
                self._t_pausa  = datetime.now()
                self.after(1500, self._mostrar_numpad)
                return

        self._en_pausa = True
        self._t_pausa  = datetime.now()

    def _foto(self, frame, prefijo) -> str:
        os.makedirs(EVIDENCIAS_DIR, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ruta = os.path.join(EVIDENCIAS_DIR, f"{prefijo}_{ts}.jpg")
        cv2.imwrite(ruta, frame)
        self._limpiar_evidencias_antiguas()
        return ruta

    def _limpiar_evidencias_antiguas(self, dias=30):
        import threading as _t
        def _limpiar():
            try:
                import time as _time
                limite = _time.time() - dias * 86400
                for archivo in os.listdir(EVIDENCIAS_DIR):
                    ruta = os.path.join(EVIDENCIAS_DIR, archivo)
                    if os.path.isfile(ruta) and os.path.getmtime(ruta) < limite:
                        os.remove(ruta)
            except Exception as e:
                print(f"[EVIDENCIAS] Error limpieza: {e}")
        _t.Thread(target=_limpiar, daemon=True).start()

    # ── Construcción UI ───────────────────────────────────────────────────────

    def _build_header(self):
        f = ctk.CTkFrame(self, fg_color=C_FRAME, corner_radius=0, height=H_HEADER)
        f.pack(fill="x"); f.pack_propagate(False)
        fl = ctk.CTkFrame(f, fg_color="transparent")
        fl.pack(side="left", padx=18, pady=0, fill="y")

        from PIL import Image as _PILImage
        _logo_pil = _PILImage.open("assets/icono.png").resize((44, 44), _PILImage.LANCZOS)
        self._logo_img = ctk.CTkImage(light_image=_logo_pil, dark_image=_logo_pil, size=(44, 44))
        lbl_logo = ctk.CTkLabel(fl, image=self._logo_img, text="",
                                cursor="arrow", fg_color="transparent")
        lbl_logo.pack(side="left", padx=(0, 12), anchor="center")
        lbl_logo.bind("<Button-1>", self._logo_click)

        fn = ctk.CTkFrame(fl, fg_color="transparent")
        fn.pack(side="left", anchor="center")
        lbl_n = ctk.CTkLabel(fn, text="VisionID", font=("Helvetica", 17, "bold"),
                              text_color=C_TXT, cursor="arrow")
        lbl_n.pack(anchor="w")
        lbl_n.bind("<Button-1>", self._logo_click)
        lbl_s = ctk.CTkLabel(fn, text="Control de Acceso", font=("Helvetica", 12),
                              text_color=C_TXT2, cursor="arrow")
        lbl_s.pack(anchor="w")
        lbl_s.bind("<Button-1>", self._logo_click)

        fr = ctk.CTkFrame(f, fg_color="transparent")
        fr.pack(side="right", padx=18, fill="y")
        self.lbl_hora = ctk.CTkLabel(fr, text="", font=("Helvetica", 26, "bold"),
                                      text_color=C_TXT)
        self.lbl_hora.pack(anchor="e", pady=(14, 0))
        self.lbl_fecha = ctk.CTkLabel(fr, text="", font=("Helvetica", 12),
                                       text_color=C_TXT2)
        self.lbl_fecha.pack(anchor="e")

    def _logo_click(self, event=None):
        """Triple clic en logo = acceso admin oculto."""
        self._logo_clicks += 1
        if self._logo_timer:
            self.after_cancel(self._logo_timer)
        if self._logo_clicks >= 3:
            self._logo_clicks = 0
            self._logo_timer  = None
            self._abrir_login()
        else:
            self._logo_timer = self.after(2000, self._logo_reset)

    def _logo_reset(self):
        self._logo_clicks = 0
        self._logo_timer  = None

    def _build_saludo(self):
        f = ctk.CTkFrame(self, fg_color=C_FOOT, corner_radius=0, height=H_SALUDO)
        f.pack(fill="x"); f.pack_propagate(False)
        self.lbl_saludo = ctk.CTkLabel(f, text="", font=("Helvetica", 13),
                                        text_color=C_TXT2)
        self.lbl_saludo.pack(side="left", padx=18)
        fc = ctk.CTkFrame(f, fg_color="transparent")
        fc.pack(side="right", padx=18)
        ctk.CTkLabel(fc, text="●", font=("Helvetica", 9),
                     text_color=C_OK).pack(side="left", padx=(0, 4))
        self.lbl_cnt_in = ctk.CTkLabel(fc, text="0 entradas", font=("Helvetica", 13),
                                        text_color=C_OK)
        self.lbl_cnt_in.pack(side="left")

    def _build_video(self):
        self.frame_video = ctk.CTkFrame(self, fg_color="#080F16",
                                         corner_radius=0, height=H_VIDEO)
        self.frame_video.pack(fill="both", expand=True)
        self.frame_video.pack_propagate(False)

        self.lbl_video = ctk.CTkLabel(self.frame_video, text="Iniciando cámara...",
                                       font=("Helvetica", 14), text_color=C_TXT2)
        self.lbl_video.place(relx=0, rely=0, relwidth=1, relheight=1)

        self._badge_frame = ctk.CTkFrame(self.frame_video, fg_color=C_FRAME,
                                          corner_radius=10, bg_color="#080F16")
        self._badge_frame.place(relx=1.0, rely=0.0, anchor="ne", x=-12, y=12)
        self.lbl_badge = ctk.CTkLabel(self._badge_frame, text="● Escaneando",
                                       font=("Helvetica", 11, "bold"),
                                       text_color=C_OK, fg_color="transparent",
                                       corner_radius=0, padx=12, pady=4)
        self.lbl_badge.pack()

        self._inst_frame = ctk.CTkFrame(self.frame_video, fg_color="#0D1E2D",
                                         corner_radius=20, bg_color="#080F16")
        self.lbl_inst = ctk.CTkLabel(self._inst_frame, text="",
                                      font=("Helvetica", 14, "bold"),
                                      text_color=C_OK, fg_color="transparent",
                                      corner_radius=0, padx=20, pady=8)
        self.lbl_inst.pack()

    def _build_overlays(self):
        self._build_msg()
        self._build_alerta_duplicado()
        self._build_numpad()
        self._build_ov_login()
        self._build_ov_registro()
        self._build_ov_captura()
        self._build_panel_admin()

    # ── Overlay de mensaje ────────────────────────────────────────────────────

    def _build_msg(self):
        self.ov_msg = ctk.CTkFrame(self.frame_video, corner_radius=0,
                                    fg_color="#0D1E2D", height=48)
        self.ov_msg.pack_propagate(False)

        self.lbl_msg_icono = ctk.CTkLabel(self.ov_msg, text="",
                                           font=("Helvetica", 20),
                                           fg_color="transparent", width=36)
        self.lbl_msg_icono.place(relx=0, rely=0.5, anchor="w", x=12)

        self._msg_texts = ctk.CTkFrame(self.ov_msg, fg_color="transparent")
        self._msg_texts.place(relx=0, rely=0.5, anchor="w", x=52)

        self.lbl_msg_titulo = ctk.CTkLabel(self._msg_texts, text="",
                                            font=("Helvetica", 13, "bold"),
                                            text_color=C_TXT, fg_color="transparent")
        self.lbl_msg_titulo.pack(anchor="w")

        self.lbl_msg_nombre = ctk.CTkLabel(self._msg_texts, text="",
                                            font=("Helvetica", 11),
                                            text_color=C_TXT2, fg_color="transparent")
        self.lbl_msg_nombre.pack(anchor="w")

        self.lbl_msg_info = ctk.CTkLabel(self.ov_msg, text="",
                                          font=("Helvetica", 10),
                                          text_color=C_TXT3, fg_color="transparent")
        self.lbl_msg_info.place(relx=1.0, rely=0.3, anchor="e", x=-12)

        self.prog_msg = ctk.CTkProgressBar(self.ov_msg, height=3, corner_radius=0,
                                            fg_color=C_BORDE, progress_color=C_OK)
        self.prog_msg.place(relx=0, rely=1.0, anchor="sw", relwidth=1)
        self.prog_msg.set(1.0)

    def _mostrar_msg(self, tipo, nombre="", matricula="", rol=""):
        dt_str = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
        if tipo == "exito":
            nom = nombre.split()[0] if nombre else "Usuario"
            icono, titulo, sub, color = (
                "✅", f"{self._saludo_hora()}, {nom}",
                f"{matricula}  ·  {rol}" if matricula else "", C_OK)
        elif tipo == "salida":
            nom = nombre.split()[0] if nombre else "Usuario"
            icono, titulo, sub, color = (
                "👋", f"Hasta luego, {nom}",
                f"{matricula}  ·  {rol}" if matricula else "", C_WARN)
        else:
            icono, titulo, sub, color = (
                "⛔", "Acceso denegado", "Rostro no registrado", C_ERROR)
        self.lbl_msg_icono.configure(text=icono)
        self.lbl_msg_titulo.configure(text=titulo, text_color=color)
        self.lbl_msg_nombre.configure(text=sub)
        self.lbl_msg_info.configure(text=dt_str)
        self.prog_msg.configure(progress_color=color)
        self.prog_msg.set(1.0)
        self.ov_msg.pack(side="bottom", fill="x")
        self.ov_msg.lift()
        self._cancelar_barra_pausa = False
        self._animar_barra_pausa(PAUSA_SEG * 1000)

    def _ocultar_msg(self):
        self._cancelar_barra_pausa = True
        self.ov_msg.pack_forget()

    def _animar_barra_pausa(self, ms_total, paso=0):
        if getattr(self, "_cancelar_barra_pausa", False):
            return
        pasos = 40; iv = ms_total // pasos
        self.prog_msg.set(max(1.0 - paso/pasos, 0.0))
        if paso < pasos:
            self.after(iv, lambda: self._animar_barra_pausa(ms_total, paso+1))

    # ── Alerta duplicado ──────────────────────────────────────────────────────

    def _build_alerta_duplicado(self):
        self.ov_duplicado = ctk.CTkFrame(self.frame_video, corner_radius=0,
                                          fg_color="#0D1E2D")
        self.ov_duplicado.pack_propagate(False)
        inner = ctk.CTkFrame(self.ov_duplicado, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(inner, text="⛔", font=("Helvetica", 52),
                     fg_color="transparent").pack(pady=(0, 8))
        ctk.CTkLabel(inner, text="ROSTRO DUPLICADO",
                     font=("Helvetica", 18, "bold"),
                     text_color=C_ERROR, fg_color="transparent").pack(pady=(0, 8))
        self.lbl_dup_nombre = ctk.CTkLabel(inner, text="",
                                            font=("Helvetica", 13, "bold"),
                                            text_color=C_TXT, fg_color="transparent")
        self.lbl_dup_nombre.pack(pady=(0, 8))
        ctk.CTkLabel(inner, text="Este rostro ya está registrado",
                     font=("Helvetica", 11),
                     text_color=C_TXT2, fg_color="transparent").pack(pady=(0, 16))
        self.prog_dup = ctk.CTkProgressBar(inner, width=200, height=4,
                                            corner_radius=2, fg_color=C_BORDE,
                                            progress_color=C_ERROR)
        self.prog_dup.pack()
        self.prog_dup.set(1.0)

    def _mostrar_alerta_duplicado(self, nombre_usuario):
        self.lbl_dup_nombre.configure(text=nombre_usuario)
        self.prog_dup.set(1.0)
        self.ov_duplicado.pack(side="bottom", fill="x")
        self.ov_duplicado.lift()
        self._cancelar_barra_duplicado = False
        self._animar_barra_duplicado(4000)

    def _ocultar_alerta_duplicado(self):
        self._cancelar_barra_duplicado = True
        if hasattr(self, "ov_duplicado"):
            self.ov_duplicado.pack_forget()

    def _animar_barra_duplicado(self, ms_total, paso=0):
        if getattr(self, "_cancelar_barra_duplicado", False):
            return
        pasos = 40; iv = ms_total // pasos
        self.prog_dup.set(max(1.0 - paso/pasos, 0.0))
        if paso < pasos:
            self.after(iv, lambda: self._animar_barra_duplicado(ms_total, paso+1))

    # ── Numpad táctil ─────────────────────────────────────────────────────────

    def _build_numpad(self):
        self._np_mayus = True
        self.ov_numpad = ctk.CTkFrame(self.frame_video, fg_color="#080F16", corner_radius=0)

        ctk.CTkLabel(self.ov_numpad, text="Acceso manual",
                     font=("Helvetica", 18, "bold"), text_color=C_TXT).pack(pady=(20, 2))
        ctk.CTkLabel(self.ov_numpad, text="Ingresa tu matrícula",
                     font=("Helvetica", 13), text_color=C_TXT2).pack()
        self.lbl_np = ctk.CTkLabel(self.ov_numpad, text="",
                                    font=("Helvetica", 26, "bold"), text_color=C_TXT,
                                    fg_color=C_FRAME, corner_radius=10, width=420, height=54)
        self.lbl_np.pack(pady=(6, 4), padx=16)

        fa = ctk.CTkFrame(self.ov_numpad, fg_color="transparent")
        fa.pack(pady=(8, 10))

        self._np_kb_frame = ctk.CTkFrame(self.ov_numpad, fg_color="transparent")
        self._np_kb_frame.pack(padx=8, fill="x", expand=True)
        self._np_botones = {}
        self._np_renderizar_teclado()


        ctk.CTkButton(fa, text="Cancelar", width=130, height=self._KB_ACT_H,
                       fg_color="transparent", text_color=C_TXT2, hover_color=C_FRAME,
                       font=("Helvetica", 13),
                       command=lambda: self._confirmar_cancelar(
                           "¿Cancelar acceso manual y volver al escaneo?",
                           accion_si=self._np_cancelar)).pack(side="left", padx=8)
        ctk.CTkButton(fa, text="OK  ✓", width=180, height=self._KB_ACT_H,
                       fg_color=C_OK, text_color=C_BG, hover_color="#00A88A",
                       font=("Helvetica", 15, "bold"), corner_radius=10,
                       command=self._np_ok).pack(side="left", padx=8)

    def _np_renderizar_teclado(self):
        for w in self._np_kb_frame.winfo_children():
            w.destroy()
        self._np_botones.clear()

        BW = self._KB_BW; BH = self._KB_BH; PAD = self._KB_PAD; FS = self._KB_FS
        filas = [
                ["1","2","3","4","5","6","7","8","9","0"],
                ["Q","W","E","R","T","Y","U","I","O","P"],
                ["A","S","D","F","G","H","J","K","L","⌫"],  # ← ⌫ aquí
                ["⇧","Z","X","C","V","B","N","M","-","_"],
       ]   

        for r_idx, fila in enumerate(filas):
            for c_idx, tecla in enumerate(fila):
                texto = tecla
                if tecla.isalpha():
                    texto = tecla if self._np_mayus else tecla.lower()

                if tecla == "⌫":
                    btn = ctk.CTkButton(self._np_kb_frame, text=tecla,
                                        width=BW+10, height=BH, font=("Helvetica", FS),
                                        fg_color=C_FRAME, text_color=C_ERROR,
                                        hover_color=C_BORDE, border_width=1,
                                        border_color=C_BORDE, corner_radius=8,
                                        command=self._np_del)
                elif tecla == "⇧":
                    btn = ctk.CTkButton(self._np_kb_frame, text=tecla,
                                        width=BW+10, height=BH, font=("Helvetica", FS),
                                        fg_color=C_OK if self._np_mayus else C_FRAME,
                                        text_color=C_BG if self._np_mayus else C_TXT,
                                        hover_color=C_BORDE, border_width=1,
                                        border_color=C_BORDE, corner_radius=8,
                                        command=self._np_toggle_mayus)
                else:
                    btn = ctk.CTkButton(self._np_kb_frame, text=texto,
                                        width=BW, height=BH,
                                        font=("Helvetica", FS, "bold"),
                                        fg_color=C_FRAME, text_color=C_TXT,
                                        hover_color=C_BORDE, border_width=1,
                                        border_color=C_BORDE, corner_radius=8,
                                        command=lambda t=texto: self._np_press(t))
                btn.grid(row=r_idx, column=c_idx, padx=PAD, pady=PAD)
                self._np_botones[tecla] = btn

    def _np_toggle_mayus(self):
        self._np_mayus = not self._np_mayus
        self._np_renderizar_teclado()

    def _mostrar_numpad(self):
        self._np_val = ""; self.lbl_np.configure(text="")
        self._en_pausa = True; self._np_vis = True
        self._esperando_numpad = False
        self._ocultar_msg()
        self.ov_numpad.place(relx=0, rely=0, relwidth=1, relheight=1)

    def _np_press(self, t):
        if len(self._np_val) < 15:
            self._np_val += t; self.lbl_np.configure(text=self._np_val)

    def _np_del(self):
        self._np_val = self._np_val[:-1]
        self.lbl_np.configure(text=self._np_val or "")

    def _np_ok(self):
        mat = self._np_val
        self.ov_numpad.place_forget(); self._np_vis = False
        self._en_pausa = False; self.fallos = 0
        u = obtener_usuario_por_matricula(mat)
        if u:
            frame = self._ultimo_frame if self._ultimo_frame is not None \
                else np.zeros((100, 100, 3), dtype=np.uint8)
            if u["id_rol"] == ROL_ALUMNO:
                grado = (u["grado"] or "").strip()
                grupo = (u["grupo"] or "").strip()
                rol_txt = f"{grado}  {grupo}".strip() if (grado or grupo) else u["nombre_rol"]
            else:
                rol_txt = u["nombre_rol"]
            id_ac = registrar_entrada(u["id_usuario"], "MANUAL")
            foto  = self._foto(frame, f"manual_{u['id_usuario']}")
            guardar_evidencia(id_ac, foto, "Entrada manual")
            self.cnt_in += 1; self._upd_cnt()
            self._set_estado("exito")
            self._mostrar_msg("exito", f"{u['nombre']} {u['apellido_p']}",
                              u["matricula"], rol_txt)
        else:
            registrar_intento_fallido(
                f"Acceso manual — matrícula no encontrada: {mat}", matricula=mat)
            self._set_estado("denegado"); self._mostrar_msg("denegado")
        self._en_pausa = True; self._t_pausa = datetime.now()

    def _np_cancelar(self):
        self.ov_numpad.place_forget(); self._np_vis = False; self._en_pausa = False
        self.fallos = 0; self._esperando_numpad = False
        self._buffer = []; self._frames_desc = 0
        self._set_estado("escaneando")

    # ── Login administrativo ──────────────────────────────────────────────────
    def _build_ov_login(self):
        self.ov_login = ctk.CTkFrame(self.frame_video, fg_color="#080F16", corner_radius=0)

        self.ov_login_form = ctk.CTkFrame(self.ov_login, fg_color="transparent")
        self.ov_login_form.pack(fill="both", expand=True)

        ctk.CTkLabel(self.ov_login_form, text="Acceso administrativo",
                     font=("Helvetica", 16, "bold"), text_color=C_TXT).pack(pady=(40, 6))
        ctk.CTkLabel(self.ov_login_form,
                     text="Ingresa tus credenciales y\nacerca tu rostro para confirmar.",
                     font=("Helvetica", 12), text_color=C_TXT2,
                     justify="center").pack(pady=(0, 24))

        self.entry_mat_l = ctk.CTkEntry(self.ov_login_form, width=300, height=44,
                                         placeholder_text="Matrícula", font=("Helvetica", 14))
        self.entry_mat_l.pack(pady=8)
        self.entry_mat_l.bind("<FocusIn>", lambda e: self._teclado.abrir(self.entry_mat_l))

        _fp = ctk.CTkFrame(self.ov_login_form, fg_color="transparent")
        _fp.pack(pady=8)
        self.entry_pass_l = ctk.CTkEntry(_fp, width=256, height=44,
                                          placeholder_text="Contraseña", show="*",
                                          font=("Helvetica", 14))
        self.entry_pass_l.pack(side="left")
        self.entry_pass_l.bind("<FocusIn>", lambda e: self._teclado.abrir(self.entry_pass_l))
        self._pass_vis_l = False
        def _toggle_pass_l():
            self._pass_vis_l = not self._pass_vis_l
            self.entry_pass_l.configure(show="" if self._pass_vis_l else "*")
        ctk.CTkButton(_fp, text="👁", width=40, height=44,
                       fg_color=C_FRAME, hover_color=C_BORDE,
                       text_color=C_TXT2, font=("Helvetica", 16),
                       command=_toggle_pass_l).pack(side="left", padx=(4, 0))

        self.lbl_login_msg = ctk.CTkLabel(self.ov_login_form, text="",
                                           font=("Helvetica", 12), text_color=C_WARN)
        self.lbl_login_msg.pack(pady=6)

        self.btn_login_confirmar = ctk.CTkButton(
            self.ov_login_form, text="Confirmar con rostro", width=260, height=46,
            font=("Helvetica", 14, "bold"), fg_color=C_OK, text_color=C_BG,
            hover_color="#00A88A", corner_radius=12, command=self._login_confirmar)
        self.btn_login_confirmar.pack(pady=10)

        ctk.CTkButton(self.ov_login_form, text="Cancelar", fg_color="transparent",
                       text_color=C_TXT2, hover_color=C_FRAME, font=("Helvetica", 12),
                       command=lambda: self._confirmar_cancelar(
                           "¿Cancelar el acceso administrativo?")).pack(pady=(4, 0))

        self.ov_login_validando = ctk.CTkFrame(self.ov_login, fg_color="transparent")
        ctk.CTkLabel(self.ov_login_validando, text="Validación de identidad",
                     font=("Helvetica", 16, "bold"), text_color=C_TXT).pack(pady=(20, 6))
        ctk.CTkLabel(self.ov_login_validando, text="Acerca tu rostro a la cámara",
                     font=("Helvetica", 13), text_color=C_TXT2).pack(pady=(0, 16))
        self.lbl_login_video = ctk.CTkLabel(self.ov_login_validando, text="",
                                             fg_color="#0A1520", corner_radius=8)
        self.lbl_login_video.pack(padx=10, pady=8)
        self.lbl_login_status = ctk.CTkLabel(self.ov_login_validando, text="",
                                              font=("Helvetica", 12, "bold"), text_color=C_OK)
        self.lbl_login_status.pack(pady=8)
        ctk.CTkButton(self.ov_login_validando, text="Cancelar", fg_color="transparent",
                       text_color=C_TXT2, hover_color=C_FRAME, font=("Helvetica", 12),
                       command=lambda: self._confirmar_cancelar(
                           "¿Cancelar la validación de identidad?")).pack(pady=(10, 0))

    def _abrir_login(self):
        self._modo = "login"
        self._en_pausa = True
        self._login_validando = False
        self._login_frames_confirmados = 0
        self._login_frames_fallidos    = 0
        while not self._queue_frames.empty():
            try: self._queue_frames.get_nowait()
            except: break
        self.entry_mat_l.delete(0, "end")
        self.entry_pass_l.delete(0, "end")
        self.lbl_login_msg.configure(text="")
        self.btn_login_confirmar.configure(state="normal")
        self._ocultar_overlays()
        self.ov_login_validando.pack_forget()
        self.ov_login_form.pack(fill="both", expand=True)
        self.ov_login.place(relx=0, rely=0, relwidth=1, relheight=1)

    def _login_confirmar(self):
        self.btn_login_confirmar.configure(state="disabled")
        mat  = self.entry_mat_l.get().strip()
        cont = self.entry_pass_l.get().strip()
        if not mat or not cont:
            self.lbl_login_msg.configure(text="Ingresa matrícula y contraseña.",
                                          text_color=C_WARN)
            self.btn_login_confirmar.configure(state="normal"); return
        u = login(mat, cont)
        if not u:
            self.lbl_login_msg.configure(text="Credenciales incorrectas.",
                                          text_color=C_ERROR)
            registrar_intento_fallido("Credenciales incorrectas login", matricula=mat)
            self.btn_login_confirmar.configure(state="normal"); return
        if not puede_registrar(u["id_rol"]):
            self.lbl_login_msg.configure(text="Tu rol no tiene permisos.",
                                          text_color=C_ERROR)
            self.btn_login_confirmar.configure(state="normal"); return
        self._login_usuario = dict(u)
        self.ov_login_form.pack_forget()
        self.ov_login_validando.pack(fill="both", expand=True)
        self.lbl_login_status.configure(text="● Detectando rostro...", text_color=C_OK)
        self._login_validando = True
        self.after(300, lambda: self._validar_rostro_login_con_video(mat))

    def _validar_rostro_login_con_video(self, matricula):
        if not self._login_validando or self._modo != "login":
            return

        frame = self._camara.leer()
        if frame is None:
            self.after(50, lambda: self._validar_rostro_login_con_video(matricula)); return

        try:
            h_orig, w_orig = frame.shape[:2]
            target_w = 280
            target_h = int(280 * h_orig / w_orig)
            if target_h > 300:
                target_h = 300
                target_w = int(300 * w_orig / h_orig)
            fd = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((300, 280, 3), dtype=np.uint8)
            canvas.fill(10)
            y_off = (300 - target_h) // 2
            x_off = (280 - target_w) // 2
            canvas[y_off:y_off+target_h, x_off:x_off+target_w] = fd
            img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            ci  = ctk.CTkImage(light_image=img, dark_image=img, size=(280, 300))
            self.lbl_login_video.configure(image=ci, text="")
            self._ci_login = ci
        except Exception as e:
            print(f"[VIDEO LOGIN] {e}")

        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small  = cv2.resize(gray, (0, 0), fx=ESCALA_DETEC, fy=ESCALA_DETEC)
        coords = self._detectar(small)

        if coords is None:
            self._login_frames_confirmados = 0
            self.lbl_login_status.configure(text="⚠ No se detectó rostro",
                                             text_color=C_WARN)
            self.after(50, lambda: self._validar_rostro_login_con_video(matricula)); return

        x, y, w, h = coords
        rostro_gray = small[y:y+h, x:x+w]
        if rostro_gray.size > 0:
        
            id_u, confianza = buscar(rostro_gray, self._recognizer)
        else:
            id_u, confianza = None, 999

        frames_fallo = 5
        if id_u != self._login_usuario["id_usuario"]:
            self._login_frames_confirmados = 0
            self._login_frames_fallidos   += 1
            self.lbl_login_status.configure(
                text=f"⚠ Validando... fallo [{self._login_frames_fallidos}/{frames_fallo}]",
                text_color=C_WARN)
            if self._login_frames_fallidos >= frames_fallo:
                self.lbl_login_status.configure(text="✗ Rostro no coincide",
                                                 text_color=C_ERROR)
                registrar_intento_fallido("Rostro no coincide en login",
                                           matricula=matricula,
                                           id_usuario=self._login_usuario["id_usuario"])
                self._login_validando = False
                self.after(2000, self._resetear_login_form); return
            self.after(50, lambda: self._validar_rostro_login_con_video(matricula)); return

        self._login_frames_confirmados += 1
        frames_necesarios = 5
        self.lbl_login_status.configure(
            text=f"✓ Validando... [{self._login_frames_confirmados}/{frames_necesarios}]",
            text_color=C_OK)
        if self._login_frames_confirmados < frames_necesarios:
            self.after(50, lambda: self._validar_rostro_login_con_video(matricula)); return

        self.lbl_login_status.configure(text="✓ Identidad confirmada", text_color=C_OK)
        print(f"[LOGIN] Acceso concedido: {self._login_usuario['nombre']} (conf: {confianza:.1f})")
        self._login_validando = False
        self.after(800, lambda: self._abrir_panel_admin(self._login_usuario))

    def _resetear_login_form(self):
        self._login_frames_confirmados = 0
        self._login_frames_fallidos    = 0
        self.ov_login_validando.pack_forget()
        self.ov_login_form.pack(fill="both", expand=True)
        self.entry_mat_l.delete(0, "end")
        self.entry_pass_l.delete(0, "end")
        self.lbl_login_msg.configure(text="")
        self.btn_login_confirmar.configure(state="normal")

    def _cancelar_modo(self):
        modo_anterior = self._modo
        self._ocultar_overlays()
        self._modo    = "acceso"
        self._en_pausa = False
        self._np_vis  = False
        self._buffer  = []
        self._frames_desc   = 0
        self._coincidencias = []
        self._login_validando          = False
        self._login_frames_confirmados = 0
        self._set_estado("escaneando")
        self._ocultar_msg()
        if modo_anterior == "login":
            self._resetear_login_form()
        elif modo_anterior == "captura":
            self._loop_camara()
        self._loop_logica()

    # ── Registro de usuarios nuevos ───────────────────────────────────────────
    def _build_ov_registro(self):
        self.ov_registro = ctk.CTkFrame(self.frame_video, fg_color="#0F1923", corner_radius=0)
        ctk.CTkLabel(self.ov_registro, text="Registrar nuevo usuario",
                     font=("Helvetica", 15, "bold"), text_color=C_TXT).pack(pady=(15, 2))
        self.lbl_reg_op = ctk.CTkLabel(self.ov_registro, text="",
                                        font=("Helvetica", 10), text_color=C_OK)
        self.lbl_reg_op.pack(pady=(0, 10))

        self._entries = {}
        form_grid = ctk.CTkFrame(self.ov_registro, fg_color="transparent")
        form_grid.pack(pady=5)

        campos = [
            ("Nombre(s)", "nombre"),            ("Apellido paterno", "apellido_p"),
            ("Apellido materno", "apellido_m"), ("Matrícula", "matricula"),
            ("Contraseña", "contrasenia"),      ("Confirmar contraseña", "contrasenia2"),
            ("Grado", "grado"),                 ("Grupo", "grupo"),
        ]
        self._grado_frame = None
        self._grupo_frame = None

        for i, (lbl, key) in enumerate(campos):
            row = i // 2; col = i % 2
            f = ctk.CTkFrame(form_grid, fg_color="transparent")
            f.grid(row=row, column=col, padx=25, pady=4, sticky="w")
            if key == "grado":
                self._grado_frame = f; f.grid_remove()
            elif key == "grupo":
                self._grupo_frame = f; f.grid_remove()
            ctk.CTkLabel(f, text=lbl, font=("Helvetica", 10),
                          text_color=C_TXT2).pack(anchor="w")

            if key in ("contrasenia", "contrasenia2"):
                fp = ctk.CTkFrame(f, fg_color="transparent")
                fp.pack(anchor="w")
                e = ctk.CTkEntry(fp, width=108, height=32, font=("Helvetica", 12), show="*")
                e.pack(side="left")
                e.bind("<FocusIn>", lambda ev, entry=e: self._teclado.abrir(entry))
                vis = [False]
                def _toggle(en=e, v=vis):
                    v[0] = not v[0]
                    en.configure(show="" if v[0] else "*")
                ctk.CTkButton(fp, text="👁", width=28, height=32,
                               fg_color=C_FRAME, hover_color=C_BORDE,
                               text_color=C_TXT2, font=("Helvetica", 13),
                               command=_toggle).pack(side="left", padx=(2, 0))
            else:
                e = ctk.CTkEntry(f, width=140, height=32, font=("Helvetica", 12))
                e.pack()
                e.bind("<FocusIn>", lambda ev, entry=e: self._teclado.abrir(entry))

            self._entries[key] = e

        rol_frame = ctk.CTkFrame(form_grid, fg_color="transparent")
        rol_frame.grid(row=4, column=0, padx=12, pady=2, sticky="w")
        ctk.CTkLabel(rol_frame, text="Rol", font=("Helvetica", 10),
                      text_color=C_TXT2).pack(anchor="w")
        self.combo_rol = ctk.CTkComboBox(rol_frame, width=140, height=32,
                                          font=("Helvetica", 12),
                                          values=["ALUMNO", "PERSONAL_ESCOLAR"],
                                          command=self._actualizar_campos_rol)
        self.combo_rol.pack()
        self.combo_rol.set("ALUMNO")

        self.lbl_reg_err = ctk.CTkLabel(self.ov_registro, text="",
                                         font=("Helvetica", 10), text_color=C_ERROR)
        self.lbl_reg_err.pack(pady=2)

        fb = ctk.CTkFrame(self.ov_registro, fg_color="transparent"); fb.pack(pady=10)
        ctk.CTkButton(fb, text="Continuar →", width=150, height=40, fg_color=C_OK,
                       text_color=C_BG, hover_color="#00A88A",
                       font=("Helvetica", 13, "bold"),
                       command=self._reg_continuar).pack(side="left", padx=6)
        ctk.CTkButton(fb, text="Cancelar", width=100, height=40,
                       fg_color="transparent", text_color=C_TXT2,
                       hover_color=C_FRAME, font=("Helvetica", 12),
                       command=lambda: self._confirmar_cancelar(
                           "¿Cancelar el registro? Se perderán los datos ingresados.")
                       ).pack(side="left", padx=6)

    def _actualizar_campos_rol(self, rol_seleccionado):
        if rol_seleccionado == "ALUMNO":
            if self._grado_frame: self._grado_frame.grid()
            if self._grupo_frame: self._grupo_frame.grid()
        else:
            if self._grado_frame: self._grado_frame.grid_remove()
            if self._grupo_frame: self._grupo_frame.grid_remove()

    def _abrir_registro(self, operador):
        self._modo = "registro"
        mapa = {1:"ADMIN", 2:"PERSONAL_AUTORIZADO", 3:"PERSONAL_ESCOLAR", 4:"ALUMNO"}
        opciones = [mapa[r] for r in roles_asignables(operador["id_rol"]) if r in mapa]
        self.combo_rol.configure(values=opciones)
        rol_inicial = opciones[-1] if opciones else "ALUMNO"
        self.combo_rol.set(rol_inicial)
        self._actualizar_campos_rol(rol_inicial)
        self.lbl_reg_op.configure(
            text=f"Operador: {operador['nombre']} {operador['apellido_p']} ({operador['nombre_rol']})")
        for e in self._entries.values(): e.delete(0, "end")
        self.lbl_reg_err.configure(text="")
        self._ocultar_overlays()
        self.ov_registro.place(relx=0, rely=0, relwidth=1, relheight=1)

    # ── Captura de rostro para registro ──────────────────────────────────────

    def _build_ov_captura(self):
        self.ov_captura = ctk.CTkFrame(self.frame_video, fg_color="#080F16", corner_radius=0)
        ctk.CTkLabel(self.ov_captura, text="Captura de rostro",
                     font=("Helvetica", 15, "bold"), text_color=C_TXT).pack(pady=(24, 2))
        self.lbl_cap_nombre = ctk.CTkLabel(self.ov_captura, text="",
                                            font=("Helvetica", 12), text_color=C_OK)
        self.lbl_cap_nombre.pack(pady=(0, 10))
        self.lbl_cap_video = ctk.CTkLabel(self.ov_captura, text="", width=320, height=240)
        self.lbl_cap_video.pack()
        ctk.CTkLabel(self.ov_captura, text="Mueve la cabeza en distintos ángulos",
                      font=("Helvetica", 11), text_color=C_TXT2).pack(pady=8)
        self.prog_cap = ctk.CTkProgressBar(self.ov_captura, width=320, height=6,
                                            corner_radius=3, fg_color=C_BORDE,
                                            progress_color=C_OK)
        self.prog_cap.pack(); self.prog_cap.set(0)
        self.lbl_cap_cnt = ctk.CTkLabel(self.ov_captura,
                                         text=f"0 / {FOTOS_CAPTURA} fotos",
                                         font=("Helvetica", 12), text_color=C_TXT2)
        self.lbl_cap_cnt.pack(pady=6)
        self.lbl_cap_instruc = ctk.CTkLabel(self.ov_captura, text="",
                                             font=("Helvetica", 11), text_color=C_WARN)
        self.lbl_cap_instruc.pack(pady=(0, 4))
        self.lbl_cap_estado = ctk.CTkLabel(self.ov_captura, text="",
                                            font=("Helvetica", 12, "bold"), text_color=C_OK)
        self.lbl_cap_estado.pack(pady=2)
        ctk.CTkButton(self.ov_captura, text="Cancelar", fg_color="transparent",
                       text_color=C_TXT2, hover_color=C_FRAME, font=("Helvetica", 12),
                       command=lambda: self._confirmar_cancelar(
                           "¿Cancelar la captura? Se perderán las fotos tomadas.")
                       ).pack(pady=(10, 0))

    def _abrir_captura(self):
        self._modo          = "captura"
        self._cap_imagenes  = []
        self._coincidencias = []
        self._cap_count     = 0
        self._etapas_captura = [
            {"nombre": "Frente",    "mensaje": "Mira al frente",                  "fotos": 10},
            {"nombre": "Izquierda", "mensaje": "Gira ligeramente a la izquierda", "fotos": 10},
            {"nombre": "Derecha",   "mensaje": "Gira ligeramente a la derecha",   "fotos": 10},
        ]
        self._etapa_actual            = 0
        self._foto_actual             = 0
        self._rostro_detectado_frames = 0
        self._posicion_valida         = False
        nom = f"{self._reg_datos['nombre']} {self._reg_datos['apellido_p']}"
        self.lbl_cap_nombre.configure(text=nom)
        self.prog_cap.set(0)
        self._actualizar_indicacion_captura()
        self._ocultar_overlays()
        self.ov_captura.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._loop_captura()

    def _actualizar_indicacion_captura(self):
        if not self._etapas_captura: return
        etapa = self._etapas_captura[self._etapa_actual]
        self.lbl_cap_estado.configure(text=etapa["mensaje"], text_color=C_OK)
        self.lbl_cap_cnt.configure(
            text=f"{self._cap_count} / {FOTOS_CAPTURA} fotos  |  "
                 f"[{self._foto_actual+1}/{etapa['fotos']}]")

    def _loop_captura(self):
        if self._modo != "captura" or self._cap_count >= FOTOS_CAPTURA:
            return

        frame = self._camara.leer()
        if frame is None:
            self.after(150, self._loop_captura); return

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
            cv2.rectangle(canvas, (marco_x, marco_y),
                          (marco_x + marco_w, marco_y + marco_h), color_marco, 3)
            img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            ci  = ctk.CTkImage(light_image=img, dark_image=img, size=(360, 390))
            self.lbl_cap_video.configure(image=ci, text="")
            self._ci_cap = ci
        except Exception as e:
            print(f"[ERROR CAPTURA] {e}")
            self.after(80, self._loop_captura); return

        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small  = cv2.resize(gray, (0, 0), fx=ESCALA_DETEC, fy=ESCALA_DETEC)
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
                centro_x = (x + w/2) / w_sm
                centro_y = (y + h/2) / h_sm
                if abs(centro_x - 0.5) > 0.15:
                    instruccion = "Centra tu rostro"
                elif abs(centro_y - 0.45) > 0.15:
                    instruccion = "Ajusta la altura"
                else:
                    rostro_bien_posicionado = True
                    instruccion = "✓ Posición correcta"

            self._posicion_valida = rostro_bien_posicionado
            if rostro_bien_posicionado:
                self._rostro_detectado_frames += 1
            else:
                self._rostro_detectado_frames = 0

            margen = 10
            x1 = max(x + margen, 0); y1 = max(y + margen, 0)
            x2 = min(x + w - margen, small.shape[1])
            y2 = min(y + h - margen, small.shape[0])
            rostro_crop = small[y1:y2, x1:x2]

            if rostro_crop.size > 0:
                rostro_res = cv2.resize(rostro_crop, FACE_SIZE)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                rostro_res = clahe.apply(rostro_res)
                rostro_res = cv2.bilateralFilter(rostro_res, 5, 75, 75)
                rostro_res = cv2.normalize(rostro_res, None, 0, 255, cv2.NORM_MINMAX)

                if self._recognizer is not None and self._cap_count >= 15 \
                        and rostro_bien_posicionado:
                    try:
                        id_existente, confianza = buscar(rostro_crop, self._recognizer)
                        if id_existente is not None and confianza < 55:
                            self._coincidencias.append(id_existente)
                        if len(self._coincidencias) >= 5:
                            id_rep = max(set(self._coincidencias),
                                         key=self._coincidencias.count)
                            if self._coincidencias.count(id_rep) >= 3:
                                u_dup = obtener_usuario_por_id(id_rep)
                                nombre_dup = (f"{u_dup['nombre']} {u_dup['apellido_p']}"
                                              if u_dup else "Usuario existente")
                                print(f"[DUPLICADO] Rostro ya registrado: {nombre_dup}")
                                self._mostrar_alerta_duplicado(nombre_dup)
                                self._modo = "cerrado_captura"  # Detiene el loop
                                self.after(4000, self._volver_a_formulario_registro); return
                    except Exception as e:
                        print(f"[VALIDACIÓN] {e}")

                if self._rostro_detectado_frames >= 3:
                    self._cap_imagenes.append(rostro_res)
                    self._foto_actual += 1
                    self._cap_count   += 1
                    self._rostro_detectado_frames = 0
                    etapa = self._etapas_captura[self._etapa_actual]
                    self.prog_cap.set(self._cap_count / FOTOS_CAPTURA)
                    self.lbl_cap_cnt.configure(
                        text=f"{self._cap_count} / {FOTOS_CAPTURA} fotos  |  "
                             f"[{self._foto_actual}/{etapa['fotos']}]",
                        text_color=C_OK)
                    if self._foto_actual >= etapa["fotos"]:
                        self._etapa_actual += 1
                        self._foto_actual   = 0
                        if self._etapa_actual >= len(self._etapas_captura):
                            self.after(150, self._finalizar_registro); return
                        else:
                            self.lbl_cap_instruc.configure(
                                text="Preparando siguiente posición...",
                                text_color=C_WARN)
                            self.lbl_cap_estado.configure(
                                text=self._etapas_captura[self._etapa_actual]["mensaje"],
                                text_color=C_OK)
                            self.after(2000, self._loop_captura); return
                    self.after(600, self._loop_captura); return
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
        """Regresa al formulario tras detectar duplicado, sin perder datos del operador."""
        self._cap_imagenes            = []
        self._cap_count               = 0
        self._coincidencias           = []
        self._etapa_actual            = 0
        self._foto_actual             = 0
        self._rostro_detectado_frames = 0
        self.prog_cap.set(0)
        self.lbl_cap_cnt.configure(text=f"0 / {FOTOS_CAPTURA} fotos")
        self._ocultar_alerta_duplicado()
        self.ov_captura.place_forget()
        self._modo = "registro"
        self._cap_imagenes = []; self._cap_count = 0
        self._coincidencias = []; self._etapa_actual = 0
        self._foto_actual = 0; self._rostro_detectado_frames = 0
        # limpiar frames acumulados
        with self._queue_frames.mutex:
            self._queue_frames.queue.clear()
        self.prog_cap.set(0)
        self.lbl_cap_cnt.configure(text=f"0 / {FOTOS_CAPTURA} fotos")
        self._ocultar_alerta_duplicado()
        self.ov_captura.place_forget()
        self.lbl_reg_err.configure(
            text="⚠ Rostro duplicado. Verifica los datos.", text_color=C_ERROR)
        # mantener datos del formulario visibles
        self.ov_registro.place(relx=0, rely=0, relwidth=1, relheight=1)
       

    def _mostrar_exito_registro(self, nom_reg):
        """Muestra pantalla de éxito sobre ov_captura con botón Listo."""
        # Overlay encima de captura
        self.ov_exito_reg = ctk.CTkFrame(
            self.frame_video, fg_color="#080F16", corner_radius=0)
        self.ov_exito_reg.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.ov_exito_reg.lift()

        inner = ctk.CTkFrame(self.ov_exito_reg, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(inner, text="✅", font=("Helvetica", 52),
                     fg_color="transparent").pack(pady=(0, 10))
        ctk.CTkLabel(inner, text="¡Registro exitoso!",
                     font=("Helvetica", 20, "bold"),
                     text_color=C_OK, fg_color="transparent").pack(pady=(0, 6))
        ctk.CTkLabel(inner, text=nom_reg,
                     font=("Helvetica", 14),
                     text_color=C_TXT, fg_color="transparent").pack(pady=(0, 4))
        ctk.CTkLabel(inner, text="El modelo se está actualizando en segundo plano.",
                     font=("Helvetica", 11),
                     text_color=C_TXT2, fg_color="transparent").pack(pady=(0, 24))

        ctk.CTkButton(inner, text="Listo  ✓", width=200, height=46,
                       fg_color=C_OK, text_color=C_BG,
                       hover_color="#00A88A",
                       font=("Helvetica", 15, "bold"), corner_radius=12,
                       command=self._listo_registro).pack()

    def _listo_registro(self):
        """Desde pantalla de éxito: si es setup inicial va a reconocimiento, si no al formulario."""
        if hasattr(self, "ov_exito_reg") and self.ov_exito_reg.winfo_exists():
            self.ov_exito_reg.place_forget()
            self.ov_exito_reg.destroy()
        # Si el operador es ADMIN setup (combo bloqueado en ADMIN sin operador logueado)
        if not self._login_usuario:
            from models.entrenadoRF import cargar_modelo_lbph
            try:
                self._recognizer = cargar_modelo_lbph()
            except Exception:
                pass
            self._admin_existe = True
            self._ocultar_overlays()
            self._modo    = "acceso"
            self._en_pausa = False
            self._set_estado("escaneando")
            self._loop_logica()
            return
        self.ov_captura.place_forget()
        self._reg_datos               = {}
        self._cap_imagenes            = []
        self._cap_count               = 0
        self._coincidencias           = []
        self._etapa_actual            = 0
        self._foto_actual             = 0
        self._rostro_detectado_frames = 0
        self.prog_cap.set(0)
        self.lbl_cap_cnt.configure(text=f"0 / {FOTOS_CAPTURA} fotos")
        self.lbl_reg_err.configure(text="")
        for e in self._entries.values():
            e.delete(0, "end")
        self.combo_rol.set("ALUMNO")
        self._actualizar_campos_rol("ALUMNO")
        self._abrir_panel_admin(self._login_usuario)

    def _validar_rostro_duplicado(self):
        if not self._cap_imagenes or self._recognizer is None:
            return False, None, 999.0
        confianza_minima = 999.0
        id_duplicado = None
        UMBRAL_DUP = 55
        for img_gray in self._cap_imagenes:
            rostro_res = cv2.resize(img_gray, FACE_SIZE)
            label, confianza = self._recognizer.predict(rostro_res)
            if confianza < UMBRAL_DUP and confianza < confianza_minima:
                confianza_minima = confianza
                id_duplicado = label
        return confianza_minima < UMBRAL_DUP, id_duplicado, confianza_minima

    def _finalizar_registro(self):
        self.lbl_cap_estado.configure(text="Procesando...", text_color=C_WARN)
        self.update()
        try:
            es_dup, id_dup, _ = self._validar_rostro_duplicado()
            if es_dup:
                u_dup = obtener_usuario_por_id(id_dup)
                nom_dup = (f"{u_dup['nombre']} {u_dup['apellido_p']}"
                           if u_dup else f"Usuario ID {id_dup}")
                self.lbl_cap_estado.configure(
                    text=f"✗ Rostro duplicado: {nom_dup}.", text_color=C_ERROR)
                self.after(3000, self._volver_a_formulario_registro); return

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

            if not self._cap_imagenes:
                raise Exception("No hay imágenes capturadas")

            for idx, img in enumerate(self._cap_imagenes):
                ruta_img = os.path.join(carpeta, f"rostro_{idx:03d}.jpg")
                if not cv2.imwrite(ruta_img, img):
                    raise Exception(f"Error escribiendo imagen {idx}")

            guardar_encoding(id_u, carpeta)

            nom_reg = f"{self._reg_datos['nombre']} {self._reg_datos['apellido_p']}"
            print(f"[BD] Registrado: {nom_reg} (ID {id_u}), {len(self._cap_imagenes)} imgs")
            self._cap_imagenes = []

            def _reentrenar():
                try:
                    ok = entrenar()
                    if ok:
                        nuevo_rec = cargar_modelo_lbph()
                        with self._lock:
                            self._recognizer = nuevo_rec
                        print(f"[LBPH] Modelo re-entrenado con {nom_reg}.")
                    else:
                        print("[LBPH] Re-entrena manualmente.")
                except Exception as e:
                    print(f"[ERROR] Re-entrenamiento: {e}")

            import threading as _t
            _t.Thread(target=_reentrenar, daemon=True).start()
            self._mostrar_exito_registro(nom_reg)

        except Exception as e:
            print(f"[ERROR REGISTRO] {e}")
            self.lbl_cap_estado.configure(text=f"Error: {e}", text_color=C_ERROR)
            self.after(3000, lambda: self.lbl_cap_estado.configure(
                text="", text_color=C_TXT2))

    # ── Estado y helpers UI ───────────────────────────────────────────────────
    def _build_panel_admin(self):
        self._panel_admin = PanelAdmin(
            parent=self.frame_video,
            on_nuevo_usuario=self._panel_ir_registro,
            on_cerrar=self._cancelar_modo,
        )

    def _panel_ir_registro(self):
        """Desde panel admin → formulario de registro."""
        self._panel_admin.cerrar()
        self._abrir_registro(self._login_usuario)

    def _abrir_panel_admin(self, operador):
        """Se llama tras login facial exitoso — muestra el panel."""
        self._modo = "panel_admin"
        self._panel_admin.abrir(operador)

    def _ocultar_overlays(self):
        for ov in [self.ov_numpad, self.ov_login, self.ov_registro, self.ov_captura]:
            ov.place_forget()
        if hasattr(self, "_panel_admin"):   # ← AGREGAR estas 2 líneas
            self._panel_admin.cerrar()
        self._ocultar_alerta_duplicado()
        self._ocultar_msg()
        if hasattr(self, "_ov_teclado") and self._ov_teclado.winfo_ismapped():
            self._ov_teclado.place_forget()

    def _set_estado(self, estado):
        self.estado = estado
        cfg = {
            "escaneando" : (C_OK,    "● Escaneando",  ""),
            "verificando": (C_OK,    "● Verificando", "Verificando identidad..."),
            "exito"      : (C_OK,    "✓ Bienvenido/a", ""),
            "salida"     : (C_WARN,  "◀ Hasta luego",  ""),
            "denegado"   : (C_ERROR, "✗ Denegado",     ""),
        }
        color, badge, inst = cfg.get(estado, (C_TXT2, "●", ""))
        self._set_badge(badge, color)
        if inst: self._set_inst(inst, color)

    def _set_badge(self, t, c):
        self.lbl_badge.configure(text=t, text_color=c)

    def _set_inst(self, t, c):
        self.lbl_inst.configure(text=t, text_color=c)
        if t:
            self._inst_frame.place(relx=0.5, rely=0.92, anchor="center")
        else:
            self._inst_frame.place_forget()

    def _upd_cnt(self):
        self.lbl_cnt_in.configure(
            text=f"{self.cnt_in} entrada{'s' if self.cnt_in != 1 else ''}")

    def _saludo_hora(self):
        h = datetime.now().hour
        if h < 12: return "Buenos días"
        if h < 19: return "Buenas tardes"
        return "Buenas noches"

    def _pulso(self):
        if self.estado == "escaneando":
            self._pulso_fase = (self._pulso_fase + 1) % 6
            self.lbl_badge.configure(
                text_color=C_OK if self._pulso_fase < 3 else C_TXT3)
        self._pulso_job = self.after(400, self._pulso)

    def _update_clock(self):
        now = datetime.now()
        # Reiniciar contador a medianoche
        if not hasattr(self, "_ultimo_dia"):
            self._ultimo_dia = now.date()
        if now.date() != self._ultimo_dia:
            self._ultimo_dia = now.date()
            self.cnt_in = 0
            self._upd_cnt()
        self.lbl_hora.configure(text=now.strftime("%H:%M:%S"))
        self.lbl_fecha.configure(text=now.strftime("%d/%m/%Y"))
        h = now.hour
        self.lbl_saludo.configure(
            text="Buenos días ☀️" if h < 12
            else "Buenas tardes 🌤" if h < 19
            else "Buenas noches 🌙")
        self.after(1000, self._update_clock)

    # ── Teclado virtual (método de compatibilidad) ────────────────────────────

    def _abrir_teclado(self, entry_target):
        """Alias para compatibilidad — usa self._teclado.abrir()"""
        self._teclado.abrir(entry_target)

    def _confirmar_cancelar(self, mensaje="¿Estás seguro que quieres cancelar?",
                         accion_si=None):
        """Overlay interno — nunca sale de la ventana de la app."""
        # Si ya hay uno abierto, cerrarlo primero
        if hasattr(self, "_ov_confirm") and self._ov_confirm.winfo_exists():
            self._ov_confirm.place_forget()
            self._ov_confirm.destroy()

        self._ov_confirm = ctk.CTkFrame(
            self.frame_video,
            fg_color=C_FRAME,
            corner_radius=16,
            border_width=2,
            border_color=C_BORDE
        )
        self._ov_confirm.place(relx=0.5, rely=0.5, anchor="center",
                                relwidth=0.82, relheight=0.35)
        self._ov_confirm.lift()

        ctk.CTkLabel(
            self._ov_confirm, text="⚠️",
            font=("Helvetica", 28), fg_color="transparent"
        ).pack(pady=(18, 4))

        ctk.CTkLabel(
            self._ov_confirm, text=mensaje,
            font=("Helvetica", 12), text_color=C_TXT,
            fg_color="transparent", wraplength=320, justify="center"
        ).pack(pady=(0, 16))

        fb = ctk.CTkFrame(self._ov_confirm, fg_color="transparent")
        fb.pack()

        def _si():
            self._ov_confirm.place_forget()
            self._ov_confirm.destroy()
            if accion_si:
                accion_si()
            else:
                self._cancelar_modo()

        def _no():
            self._ov_confirm.place_forget()
            self._ov_confirm.destroy()

        ctk.CTkButton(
            fb, text="Sí, cancelar", width=130, height=40,
            fg_color=C_ERROR, text_color="white",
            hover_color="#a00000", font=("Helvetica", 12, "bold"),
            command=_si
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            fb, text="No, continuar", width=130, height=40,
            fg_color=C_OK, text_color=C_BG,
            hover_color="#00A88A", font=("Helvetica", 12, "bold"),
            command=_no
        ).pack(side="left", padx=8)

    # ── Cierre de la aplicación ───────────────────────────────────────────────

    def _cerrar(self):
        if self._pulso_job:
            self.after_cancel(self._pulso_job)
        self._camara.liberar()
        self.destroy()


if __name__ == "__main__":
    app = FaceAccess()
    app.mainloop()