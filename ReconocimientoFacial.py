"""
ReconocimientoFacial.py 
"""

import cv2
import os
import json
import threading
import time
import numpy as np
import face_recognition
import customtkinter as ctk
from PIL import Image, ImageDraw

# Detectar si estamos en Raspberry Pi con cámara CSI
try:
    from picamera2 import Picamera2
    PICAMERA2_DISPONIBLE = True
except ImportError:
    PICAMERA2_DISPONIBLE = False
from collections import Counter
from datetime import datetime
from db_manager import (
    obtener_todos_encodings,
    obtener_usuario_por_id,
    obtener_usuario_por_matricula,
    tiene_entrada_abierta,
    registrar_entrada,
    registrar_salida,
    registrar_intento_fallido,
    guardar_evidencia,
    login,
    puede_registrar,
    roles_asignables,
    registrar_usuario,
    guardar_encoding,
)

TOLERANCIA     = 0.42   # más estricto: evita reconocer perfiles o vistas parciales
ESCALA_DETEC   = 0.25
FRAMES_CONFIRM = 3
PAUSA_SEG      = 3
FALLOS_NUMPAD  = 2
EVIDENCIAS_DIR = "evidencias"
MODELO_YUNET   = "face_detection_yunet_2023mar.onnx"
FOTOS_CAPTURA  = 10

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
C_ADMIN = "#534AB7"

H_HEADER = 58
H_SALUDO = 34
H_VIDEO  = 800 - H_HEADER - H_SALUDO



def cargar_encodings():
    datos = obtener_todos_encodings()
    enc_list, ids_list = [], []
    for id_u, enc_str in datos:
        try:
            vec = np.array(json.loads(enc_str), dtype=np.float64)
            if vec.shape == (128,):
                enc_list.append(vec)
                ids_list.append(id_u)
        except Exception:
            pass
    print(f"[INFO] {len(enc_list)} encoding(s) en RAM.")
    return enc_list, ids_list


def buscar(enc_nuevo, encodings, ids):
    if not encodings:
        return None, 1.0
    dists = face_recognition.face_distance(encodings, enc_nuevo)
    idx   = int(np.argmin(dists))
    dist  = float(dists[idx])
    return (ids[idx], dist) if dist <= TOLERANCIA else (None, dist)


class FaceAccess(ctk.CTk):

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("FaceAccess")
        self.geometry("480x800")
        self.resizable(False, False)
        self.configure(fg_color=C_BG)

        self.estado      = "escaneando"
        self.fallos      = 0
        self.cnt_in      = 0
        self.cnt_out     = 0
        self._pulso_fase = 0
        self._pulso_job  = None
        self._ci         = None
        self._np_val     = ""
        self._np_vis     = False

        self._lock          = threading.Lock()
        self._frame_rec     = None
        self._frame_nuevo   = False
        self._resultado     = None
        self._res_nuevo     = False
        self._ultimo_frame  = None
        self._en_pausa      = False
        self._t_pausa       = None
        self._buffer        = []
        self._frames_desc   = 0
        self._ultimo_coords = None
        self._ultimo_id_u   = None
        self._t_ultimo_res  = 0.0   # timestamp del último resultado del hilo

        self._encodings, self._ids = cargar_encodings()

        self._modo          = "acceso"
        self._login_usuario = None
        self._cap_encodings = []
        self._cap_count     = 0
        self._reg_datos     = {}

        self._build_header()
        self._build_saludo()
        self._build_video()
        self._build_overlays()

        self._detector = self._init_yunet()
        self._update_clock()
        self._pulso()
        self.protocol("WM_DELETE_WINDOW", self._cerrar)
        self.after(400, self._iniciar)

    def _init_yunet(self):
        if not os.path.exists(MODELO_YUNET):
            print("[MOTOR] YuNet no encontrado.")
            return None
        try:
            det = cv2.FaceDetectorYN.create(
                MODELO_YUNET, "",
                (int(640*ESCALA_DETEC), int(480*ESCALA_DETEC)),
                score_threshold=0.85,
                nms_threshold=0.3,
                top_k=1)
            print("[MOTOR] YuNet activo.")
            return det
        except Exception as e:
            print(f"[MOTOR] YuNet falló: {e}")
            return None

    def _iniciar(self):
        if PICAMERA2_DISPONIBLE:
            self._picam = Picamera2()
            config = self._picam.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"})
            self._picam.configure(config)
            self._picam.start()
            self.cap = None
            print("[CAMARA] Usando picamera2 (CSI)")
        else:
            self._picam = None
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print("[CAMARA] Usando OpenCV (USB/webcam)")
        threading.Thread(target=self._hilo_rec, daemon=True).start()
        self._loop_camara()
        self._loop_logica()

    # ── Hilo de reconocimiento ────────────────────────────────────────────

    def _hilo_rec(self):
        frame = None
        while True:
            with self._lock:
                if self._frame_nuevo:
                    frame = self._frame_rec.copy()
                    self._frame_nuevo = False
            if frame is None:
                time.sleep(0.005)
                continue

            small_bgr = cv2.resize(frame, (0, 0), fx=ESCALA_DETEC, fy=ESCALA_DETEC)
            small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
            ubs       = self._detectar(small_bgr)

            def sin_rostro():
                with self._lock:
                    self._resultado    = {"tipo": "sin_rostro"}
                    self._res_nuevo    = True
                    self._t_ultimo_res = time.monotonic()

            if not ubs:
                sin_rostro(); frame = None; continue

            top, r, b, l = ubs[0]
            h_s, w_s     = small_bgr.shape[:2]
            if (b-top)/h_s < 0.15 or (r-l)/w_s < 0.10:
                sin_rostro(); frame = None; continue

            encs = face_recognition.face_encodings(
                small_rgb, [ubs[0]], num_jitters=0, model="small")
            if not encs:
                sin_rostro(); frame = None; continue

            id_u, dist = buscar(encs[0], self._encodings, self._ids)

            # Guardar coords en el espacio del frame reducido (ESCALA_DETEC)
            # _dibujar_rect las escalará al tamaño de pantalla
            coords = (top, r, b, l)

            with self._lock:
                self._resultado = {
                    "tipo"      : "rostro",
                    "id_usuario": id_u,
                    "distancia" : dist,
                    "frame_cap" : frame.copy(),
                    "coords"    : coords,
                }
                self._res_nuevo  = True
                self._t_ultimo_res = time.monotonic()
            frame = None

    def _detectar(self, small_bgr):
        """
        Detecta rostros con YuNet.
        Validaciones:
          · Score de confianza >= 0.88
          · Nariz entre los ojos (rostro de frente, no perfil extremo)
          · Ambos ojos visibles
        """
        if not self._detector:
            return []
        h, w = small_bgr.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(small_bgr)
        if faces is None:
            return []

        f = faces[0]

        # Fix 3 — Score mínimo de confianza
        score = float(f[14]) if len(f) > 14 else 1.0
        if score < 0.88:
            return []

        # Fix 2 — Validar que el rostro esté de frente
        # YuNet landmarks: [x,y] de ojo_der, ojo_izq, nariz, boca_der, boca_izq
        # índices: 4-5=ojo_der, 6-7=ojo_izq, 8-9=nariz, 10-11=boca_der, 12-13=boca_izq
        if len(f) >= 14:
            ojo_der_x = float(f[4])
            ojo_izq_x = float(f[6])
            nariz_x   = float(f[8])

            x_min = min(ojo_der_x, ojo_izq_x)
            x_max = max(ojo_der_x, ojo_izq_x)

            # La nariz debe estar entre los dos ojos horizontalmente
            # Si está fuera → perfil extremo → ignorar
            margen = (x_max - x_min) * 0.25   # 25% de tolerancia lateral
            if not (x_min - margen < nariz_x < x_max + margen):
                return []

            # Los dos ojos deben estar separados (no son el mismo punto)
            dist_ojos = abs(ojo_izq_x - ojo_der_x)
            if dist_ojos < 8:   # píxeles en frame reducido al 25%
                return []

        return [(int(f[1]), int(f[0]+f[2]), int(f[1]+f[3]), int(f[0]))]

    # ── Loop cámara ───────────────────────────────────────────────────────

    def _loop_camara(self):
        # En modo captura la cámara la lee _loop_captura — no seguir
        if self._modo == "captura":
            return
        if self._np_vis or self._modo not in ("acceso", "login"):
            self.after(33, self._loop_camara)
            return

        # Leer frame según el tipo de cámara
        if self._picam:
            frame = self._picam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.flip(frame, 1)
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.after(33, self._loop_camara)
                return
            frame = cv2.flip(frame, 1)
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

            # Enviar al hilo el mismo frame recortado que se muestra
            # así lo que detecta coincide exactamente con lo que ve el usuario
            if not self._en_pausa and self._modo == "acceso":
                with self._lock:
                    self._frame_rec   = f.copy()
                    self._frame_nuevo = True

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
        """
        Dibuja esquinas alrededor del rostro detectado.
        Las coords ya son del frame recortado (mismo que se muestra).
        """
        with self._lock:
            coords = self._ultimo_coords
            id_u   = self._ultimo_id_u

        if coords is None or self._en_pausa:
            return frame

        top_s, r_s, b_s, l_s = coords

        # El frame que llega es tw x th
        # Las coords son del frame reducido (ESCALA_DETEC del frame mostrado)
        # Escalar UNA sola vez al tamaño real
        esc   = 1.0 / ESCALA_DETEC
        top_s = int(top_s * esc)
        b_s   = int(b_s   * esc)
        l_s   = int(l_s   * esc)
        r_s   = int(r_s   * esc)

        # Clamp
        top_s = max(top_s, 0); b_s = min(b_s, th)
        l_s   = max(l_s,   0); r_s = min(r_s, tw)

        if b_s <= top_s or r_s <= l_s:
            return frame

        # Color según estado
        colores = {
            "escaneando" : (180, 180, 180),
            "verificando": (0,   212, 170),
            "exito"      : (0,   212, 170),
            "salida"     : (35,  166, 245),
            "denegado"   : (74,  75,  226),
        }
        color = colores.get(self.estado, (180, 180, 180))

        # Rectángulo con esquinas redondeadas visualmente
        # (líneas en las 4 esquinas, no el rectángulo completo — más elegante)
        largo  = max(min(int((r_s - l_s) * 0.20), 30), 15)  # largo de cada esquina
        grosor = 3

        # Esquina superior izquierda
        cv2.line(frame, (l_s, top_s), (l_s + largo, top_s), color, grosor)
        cv2.line(frame, (l_s, top_s), (l_s, top_s + largo), color, grosor)
        # Esquina superior derecha
        cv2.line(frame, (r_s, top_s), (r_s - largo, top_s), color, grosor)
        cv2.line(frame, (r_s, top_s), (r_s, top_s + largo), color, grosor)
        # Esquina inferior izquierda
        cv2.line(frame, (l_s, b_s), (l_s + largo, b_s), color, grosor)
        cv2.line(frame, (l_s, b_s), (l_s, b_s - largo), color, grosor)
        # Esquina inferior derecha
        cv2.line(frame, (r_s, b_s), (r_s - largo, b_s), color, grosor)
        cv2.line(frame, (r_s, b_s), (r_s, b_s - largo), color, grosor)

        # Pulso: borde completo semitransparente en escaneando
        if self.estado == "escaneando":
            alpha = 0.15 + 0.12 * (self._pulso_fase / 5)
            ov    = frame.copy()
            cv2.rectangle(ov, (l_s, top_s), (r_s, b_s), color, 1)
            cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

        return frame

    # ── Loop lógica ───────────────────────────────────────────────────────

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

        with self._lock:
            # Si no hay resultado nuevo Y el último tiene más de 400ms → sin_rostro
            if not self._res_nuevo:
                edad = time.monotonic() - self._t_ultimo_res
                if edad > 0.4 and self._t_ultimo_res > 0:
                    # Forzar limpieza — el hilo no ha enviado nada reciente
                    self._buffer        = []
                    self._frames_desc   = 0
                    self._ultimo_coords = None
                    self._ultimo_id_u   = None
                    if self.estado not in ("escaneando",):
                        self.after(0, lambda: (self._set_estado("escaneando"),
                                               self._ocultar_msg()))
                self.after(100, self._loop_logica)
                return
            res = self._resultado
            self._res_nuevo = False

        if res["tipo"] == "sin_rostro":
            self._buffer      = []
            self._frames_desc = 0
            with self._lock:
                self._ultimo_coords = None
                self._ultimo_id_u   = None
            if self.estado != "escaneando":
                self._set_estado("escaneando")
                self._ocultar_msg()
            self.after(100, self._loop_logica)
            return

        id_u = res["id_usuario"]

        # Guardar coords para dibujar rectángulo
        with self._lock:
            self._ultimo_coords = res.get("coords")
            self._ultimo_id_u   = id_u

        if id_u is not None:
            self._frames_desc = 0
            self._buffer.append(id_u)
            if len(self._buffer) < FRAMES_CONFIRM:
                self._set_badge("● Verificando", C_OK)
                self.after(100, self._loop_logica)
                return
            conteo   = Counter(self._buffer)
            id_final = conteo.most_common(1)[0][0]
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
            mat    = u["matricula"]  if u else ""
            rol    = u["nombre_rol"] if u else ""

            if tiene_entrada_abierta(id_usuario):
                registrar_salida(id_usuario)
                print(f"[BD] SALIDA  — {nombre} | {dt}")
                self.cnt_out += 1; self._upd_cnt()
                self._set_estado("salida")
                self._mostrar_msg("salida", nombre, mat, rol)
            else:
                id_ac = registrar_entrada(id_usuario, "FACIAL")
                print(f"[BD] ENTRADA — {nombre} | {dt}")
                self.cnt_in += 1; self._upd_cnt()
                self._set_estado("exito")
                self._mostrar_msg("exito", nombre, mat, rol)
        else:
            registrar_intento_fallido("Rostro no reconocido")
            print(f"[BD] DENEGADO | {dt}")
            self.fallos += 1
            self._set_estado("denegado")
            self._mostrar_msg("denegado")
            if self.fallos >= FALLOS_NUMPAD:
                self.after(1500, self._mostrar_numpad)
                return

        self._en_pausa = True
        self._t_pausa  = datetime.now()

    def _foto(self, frame, prefijo) -> str:
        """Guarda foto de evidencia. Solo se llama en acceso MANUAL."""
        os.makedirs(EVIDENCIAS_DIR, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ruta = os.path.join(EVIDENCIAS_DIR, f"{prefijo}_{ts}.jpg")
        cv2.imwrite(ruta, frame)
        self._limpiar_evidencias_antiguas()
        return ruta

    def _limpiar_evidencias_antiguas(self, dias=30):
        """Elimina fotos de evidencias con más de N días. Corre en hilo."""
        import threading as _t
        def _limpiar():
            try:
                import time as _time
                limite = _time.time() - dias * 86400
                for archivo in os.listdir(EVIDENCIAS_DIR):
                    ruta = os.path.join(EVIDENCIAS_DIR, archivo)
                    if os.path.isfile(ruta) and os.path.getmtime(ruta) < limite:
                        os.remove(ruta)
                        print(f"[EVIDENCIAS] Eliminada: {archivo}")
            except Exception as e:
                print(f"[EVIDENCIAS] Error limpieza: {e}")
        _t.Thread(target=_limpiar, daemon=True).start()

    # ── Construcción UI ───────────────────────────────────────────────────

    def _build_header(self):
        f = ctk.CTkFrame(self, fg_color=C_FRAME, corner_radius=0, height=H_HEADER)
        f.pack(fill="x"); f.pack_propagate(False)
        fl = ctk.CTkFrame(f, fg_color="transparent")
        fl.pack(side="left", padx=14, pady=10)
        cv = ctk.CTkCanvas(fl, width=32, height=32, bg=C_FRAME, highlightthickness=0)
        cv.pack(side="left", padx=(0, 8))
        cv.create_oval(2, 2, 30, 30, fill=C_OK, outline="")
        cv.create_text(16, 16, text="FA", fill=C_BG, font=("Helvetica", 10, "bold"))
        fn = ctk.CTkFrame(fl, fg_color="transparent")
        fn.pack(side="left")
        ctk.CTkLabel(fn, text="VisionID", font=("Helvetica", 13, "bold"), text_color=C_TXT).pack(anchor="w")
        ctk.CTkLabel(fn, text="Control de Acceso", font=("Helvetica", 10), text_color=C_TXT2).pack(anchor="w")
        fr = ctk.CTkFrame(f, fg_color="transparent")
        fr.pack(side="right", padx=14)
        self.lbl_hora = ctk.CTkLabel(fr, text="", font=("Helvetica", 18, "bold"), text_color=C_TXT)
        self.lbl_hora.pack(anchor="e")
        self.lbl_fecha = ctk.CTkLabel(fr, text="", font=("Helvetica", 10), text_color=C_TXT2)
        self.lbl_fecha.pack(anchor="e")

    def _build_saludo(self):
        f = ctk.CTkFrame(self, fg_color=C_FOOT, corner_radius=0, height=H_SALUDO)
        f.pack(fill="x"); f.pack_propagate(False)
        self.lbl_saludo = ctk.CTkLabel(f, text="", font=("Helvetica", 11), text_color=C_TXT2)
        self.lbl_saludo.pack(side="left", padx=14)
        fc = ctk.CTkFrame(f, fg_color="transparent")
        fc.pack(side="right", padx=14)
        ctk.CTkLabel(fc, text="●", font=("Helvetica", 8), text_color=C_OK).pack(side="left", padx=(0, 3))
        self.lbl_cnt_in = ctk.CTkLabel(fc, text="0 entradas", font=("Helvetica", 10), text_color=C_OK)
        self.lbl_cnt_in.pack(side="left", padx=(0, 10))
        ctk.CTkLabel(fc, text="●", font=("Helvetica", 8), text_color=C_TXT2).pack(side="left", padx=(0, 3))
        self.lbl_cnt_out = ctk.CTkLabel(fc, text="0 salidas", font=("Helvetica", 10), text_color=C_TXT2)
        self.lbl_cnt_out.pack(side="left")

    def _build_video(self):
        self.frame_video = ctk.CTkFrame(self, fg_color="#080F16", corner_radius=0, height=H_VIDEO)
        self.frame_video.pack(fill="both", expand=True)
        self.frame_video.pack_propagate(False)
        self.lbl_video = ctk.CTkLabel(self.frame_video, text="Iniciando cámara...", font=("Helvetica", 14), text_color=C_TXT2)
        self.lbl_video.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.lbl_badge = ctk.CTkLabel(self.frame_video, text="● Escaneando", font=("Helvetica", 11, "bold"), text_color=C_OK, fg_color=C_FRAME, corner_radius=10, padx=12, pady=4)
        self.lbl_badge.place(relx=1.0, rely=0.0, anchor="ne", x=-12, y=12)
        self.lbl_inst = ctk.CTkLabel(self.frame_video, text="", font=("Helvetica", 14, "bold"), text_color=C_OK, fg_color="#0D1E2D", corner_radius=20, padx=20, pady=8)
        self.lbl_inst.place(relx=0.5, rely=0.92, anchor="center")
        self.btn_reg = ctk.CTkButton(self.frame_video, text="＋ Registrar", font=("Helvetica", 11, "bold"), width=110, height=36, fg_color=C_ADMIN, hover_color="#3C3489", text_color="white", corner_radius=10, command=self._abrir_login)
        self.btn_reg.place(relx=1.0, rely=1.0, anchor="se", x=-12, y=-12)

    def _build_overlays(self):
        self._build_msg()
        self._build_numpad()
        self._build_ov_login()
        self._build_ov_registro()
        self._build_ov_captura()

    def _build_msg(self):
        """
        Tarjeta emergente — width y height en el CONSTRUCTOR
        para cumplir con CustomTkinter.
        """
        self.ov_msg = ctk.CTkFrame(
            self.frame_video,
            corner_radius=24,
            fg_color="#0D1E2D",
            width=320,
            height=280)
        self.ov_msg.pack_propagate(False)

        self.lbl_msg_icono = ctk.CTkLabel(self.ov_msg, text="", font=("Helvetica", 52))
        self.lbl_msg_icono.pack(pady=(24, 4))
        self.lbl_msg_titulo = ctk.CTkLabel(self.ov_msg, text="", font=("Helvetica", 20, "bold"), text_color=C_TXT)
        self.lbl_msg_titulo.pack(pady=(0, 4))
        self.lbl_msg_nombre = ctk.CTkLabel(self.ov_msg, text="", font=("Helvetica", 13), text_color=C_TXT2)
        self.lbl_msg_nombre.pack(pady=(0, 4))
        self.lbl_msg_info = ctk.CTkLabel(self.ov_msg, text="", font=("Helvetica", 11), text_color=C_TXT3)
        self.lbl_msg_info.pack(pady=(0, 16))
        self.prog_msg = ctk.CTkProgressBar(self.ov_msg, width=200, height=4, corner_radius=2, fg_color=C_BORDE, progress_color=C_OK)
        self.prog_msg.pack(pady=(0, 20))
        self.prog_msg.set(1.0)

    def _mostrar_msg(self, tipo, nombre="", matricula="", rol=""):
        dt_str = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
        if tipo == "exito":
            nom = nombre.split()[0] if nombre else "Usuario"
            icono, titulo, sub, color = "✅", f"{self._saludo_hora()}, {nom}", f"{matricula}  ·  {rol}" if matricula else "", C_OK
        elif tipo == "salida":
            nom = nombre.split()[0] if nombre else "Usuario"
            icono, titulo, sub, color = "👋", f"Hasta luego, {nom}", f"{matricula}  ·  {rol}" if matricula else "", C_WARN
        else:
            icono, titulo, sub, color = "⛔", "Acceso denegado", "Rostro no registrado", C_ERROR

        self.lbl_msg_icono.configure(text=icono)
        self.lbl_msg_titulo.configure(text=titulo, text_color=color)
        self.lbl_msg_nombre.configure(text=sub)
        self.lbl_msg_info.configure(text=dt_str)
        self.prog_msg.configure(progress_color=color)
        self.prog_msg.set(1.0)
        # place SIN width/height — ya están en el constructor
        self.ov_msg.place(relx=0.5, rely=0.5, anchor="center")
        self.ov_msg.lift()
        self._cancelar_barra_pausa = False
        self._animar_barra_pausa(PAUSA_SEG * 1000)

    def _ocultar_msg(self):
        self._cancelar_barra_pausa = True
        self.ov_msg.place_forget()

    def _animar_barra_pausa(self, ms_total, paso=0):
        if getattr(self, "_cancelar_barra_pausa", False):
            return
        pasos    = 40
        iv       = ms_total // pasos
        self.prog_msg.set(max(1.0 - paso/pasos, 0.0))
        if paso < pasos:
            self.after(iv, lambda: self._animar_barra_pausa(ms_total, paso+1))

    def _build_numpad(self):
        """
        Teclado táctil completo para matrícula tipo ALU2024001.
        Filas: números | letras A-J | letras K-T | letras U-Z + especiales
        Botón ⇧ alterna mayúsculas/minúsculas.
        """
        self._np_mayus = True   # empieza en mayúsculas
        self.ov_numpad = ctk.CTkFrame(self.frame_video, fg_color="#080F16", corner_radius=0)

        ctk.CTkLabel(self.ov_numpad, text="Acceso manual",
                     font=("Helvetica", 15, "bold"),
                     text_color=C_TXT).pack(pady=(18, 2))
        ctk.CTkLabel(self.ov_numpad, text="Ingresa tu matrícula",
                     font=("Helvetica", 11), text_color=C_TXT2).pack()

        self.lbl_np = ctk.CTkLabel(
            self.ov_numpad, text="",
            font=("Helvetica", 22, "bold"), text_color=C_TXT,
            fg_color=C_FRAME, corner_radius=8, width=380, height=46)
        self.lbl_np.pack(pady=(8, 6), padx=20)

        # Frame del teclado
        self._np_kb_frame = ctk.CTkFrame(self.ov_numpad, fg_color="transparent")
        self._np_kb_frame.pack(padx=12)

        self._np_botones = {}  # key → widget botón

        self._np_renderizar_teclado()

        # Fila de acción
        fa = ctk.CTkFrame(self.ov_numpad, fg_color="transparent")
        fa.pack(pady=(6, 4))
        ctk.CTkButton(fa, text="Cancelar", width=110, height=36,
                       fg_color="transparent", text_color=C_TXT2,
                       hover_color=C_FRAME, font=("Helvetica", 11),
                       command=self._np_cancelar).pack(side="left", padx=6)
        ctk.CTkButton(fa, text="OK  ✓", width=150, height=36,
                       fg_color=C_OK, text_color=C_BG,
                       hover_color="#00A88A",
                       font=("Helvetica", 13, "bold"),
                       corner_radius=8,
                       command=self._np_ok).pack(side="left", padx=6)

    def _np_renderizar_teclado(self):
        """Construye o reconstruye el teclado según el estado mayúsculas."""
        for w in self._np_kb_frame.winfo_children():
            w.destroy()
        self._np_botones.clear()

        BW, BH = 42, 40   # ancho y alto de cada tecla
        PAD    = 3

        filas = [
            ["1","2","3","4","5","6","7","8","9","0","⌫"],
            ["Q","W","E","R","T","Y","U","I","O","P"],
            ["A","S","D","F","G","H","J","K","L"],
            ["⇧","Z","X","C","V","B","N","M","-"],
        ]

        for r_idx, fila in enumerate(filas):
            for c_idx, tecla in enumerate(fila):
                texto = tecla
                if tecla.isalpha():
                    texto = tecla if self._np_mayus else tecla.lower()

                if tecla == "⌫":
                    btn = ctk.CTkButton(
                        self._np_kb_frame, text=tecla,
                        width=BW+6, height=BH,
                        font=("Helvetica", 14),
                        fg_color=C_FRAME, text_color=C_ERROR,
                        hover_color=C_BORDE,
                        border_width=1, border_color=C_BORDE,
                        corner_radius=6, command=self._np_del)
                elif tecla == "⇧":
                    color_mayus = C_OK if self._np_mayus else C_FRAME
                    btn = ctk.CTkButton(
                        self._np_kb_frame, text=tecla,
                        width=BW+6, height=BH,
                        font=("Helvetica", 14),
                        fg_color=color_mayus,
                        text_color=C_BG if self._np_mayus else C_TXT,
                        hover_color=C_BORDE,
                        border_width=1, border_color=C_BORDE,
                        corner_radius=6, command=self._np_toggle_mayus)
                else:
                    btn = ctk.CTkButton(
                        self._np_kb_frame, text=texto,
                        width=BW, height=BH,
                        font=("Helvetica", 13, "bold"),
                        fg_color=C_FRAME, text_color=C_TXT,
                        hover_color=C_BORDE,
                        border_width=1, border_color=C_BORDE,
                        corner_radius=6,
                        command=lambda t=texto: self._np_press(t))

                btn.grid(row=r_idx, column=c_idx, padx=PAD, pady=PAD)
                self._np_botones[tecla] = btn

    def _np_toggle_mayus(self):
        self._np_mayus = not self._np_mayus
        self._np_renderizar_teclado()

    def _mostrar_numpad(self):
        self._np_val = ""; self.lbl_np.configure(text="")
        self._en_pausa = True; self._np_vis = True
        self._ocultar_msg()
        self.ov_numpad.place(relx=0, rely=0, relwidth=1, relheight=1)

    def _np_press(self, t):
        if len(self._np_val) < 15:   # matrícula hasta 15 chars
            self._np_val += t
            self.lbl_np.configure(text=self._np_val)

    def _np_del(self):
        self._np_val = self._np_val[:-1]; self.lbl_np.configure(text=self._np_val or "")

    def _np_ok(self):
        mat = self._np_val
        self.ov_numpad.place_forget(); self._np_vis = False; self._en_pausa = False; self.fallos = 0
        u = obtener_usuario_por_matricula(mat)
        if u:
            frame = self._ultimo_frame if self._ultimo_frame is not None else np.zeros((100,100,3), dtype=np.uint8)
            id_ac = registrar_entrada(u["id_usuario"], "MANUAL")
            foto  = self._foto(frame, f"manual_{u['id_usuario']}")
            guardar_evidencia(id_ac, foto, "Entrada manual")
            self.cnt_in += 1; self._upd_cnt()
            self._set_estado("exito")
            self._mostrar_msg("exito", f"{u['nombre']} {u['apellido_p']}", u["matricula"], u["nombre_rol"])
        else:
            registrar_intento_fallido(f"Acceso manual — matrícula no encontrada: {mat}", matricula=mat)
            self._set_estado("denegado"); self._mostrar_msg("denegado")
        self._en_pausa = True; self._t_pausa = datetime.now()

    def _np_cancelar(self):
        self.ov_numpad.place_forget(); self._np_vis = False; self._en_pausa = False
        self.fallos = 0; self._buffer = []; self._frames_desc = 0
        self._set_estado("escaneando")

    def _build_ov_login(self):
        self.ov_login = ctk.CTkFrame(self.frame_video, fg_color="#080F16", corner_radius=0)
        ctk.CTkLabel(self.ov_login, text="Acceso administrativo", font=("Helvetica", 16, "bold"), text_color=C_TXT).pack(pady=(40, 6))
        ctk.CTkLabel(self.ov_login, text="Ingresa tus credenciales y\nacerca tu rostro para confirmar.", font=("Helvetica", 12), text_color=C_TXT2, justify="center").pack(pady=(0, 24))
        self.entry_mat_l = ctk.CTkEntry(self.ov_login, width=300, height=44, placeholder_text="Matrícula", font=("Helvetica", 14))
        self.entry_mat_l.pack(pady=8)
        self.entry_mat_l.bind("<FocusIn>", lambda e: self._abrir_teclado(self.entry_mat_l))
        self.entry_pass_l = ctk.CTkEntry(self.ov_login, width=300, height=44, placeholder_text="Contraseña", show="*", font=("Helvetica", 14))
        self.entry_pass_l.pack(pady=8)
        self.entry_pass_l.bind("<FocusIn>", lambda e: self._abrir_teclado(self.entry_pass_l))
        self.lbl_login_msg = ctk.CTkLabel(self.ov_login, text="", font=("Helvetica", 12), text_color=C_WARN)
        self.lbl_login_msg.pack(pady=6)
        self.btn_login_confirmar = ctk.CTkButton(self.ov_login, text="Confirmar con rostro", width=260, height=46, font=("Helvetica", 14, "bold"), fg_color=C_OK, text_color=C_BG, hover_color="#00A88A", corner_radius=12, command=self._login_confirmar)
        self.btn_login_confirmar.pack(pady=10)
        ctk.CTkButton(self.ov_login, text="Cancelar", fg_color="transparent", text_color=C_TXT2, hover_color=C_FRAME, font=("Helvetica", 12), command=self._cancelar_modo).pack(pady=(4, 0))

    def _abrir_login(self):
        self._modo = "login"; self._en_pausa = True
        self.entry_mat_l.delete(0, "end"); self.entry_pass_l.delete(0, "end")
        self.lbl_login_msg.configure(text="")
        self._ocultar_overlays()
        self.ov_login.place(relx=0, rely=0, relwidth=1, relheight=1)

    def _login_confirmar(self):
        self.btn_login_confirmar.configure(state="disabled")
        mat  = self.entry_mat_l.get().strip()
        cont = self.entry_pass_l.get().strip()
        if not mat or not cont:
            self.lbl_login_msg.configure(text="Ingresa matrícula y contraseña.", text_color=C_WARN); self.btn_login_confirmar.configure(state="normal"); return
        u = login(mat, cont)
        if not u:
            self.lbl_login_msg.configure(text="Credenciales incorrectas.", text_color=C_ERROR)
            registrar_intento_fallido("Credenciales incorrectas login", matricula=mat); self.btn_login_confirmar.configure(state="normal"); return
        if not puede_registrar(u["id_rol"]):
            self.lbl_login_msg.configure(text="Tu rol no tiene permisos.", text_color=C_ERROR); self.btn_login_confirmar.configure(state="normal"); return
        self.lbl_login_msg.configure(text="Acerca tu rostro a la cámara...", text_color=C_OK)
        self._login_usuario = dict(u)
        self.after(600, lambda: self._validar_rostro_login(mat))

    def _validar_rostro_login(self, matricula):
        frame = self._ultimo_frame
        if frame is None:
            self.after(300, lambda: self._validar_rostro_login(matricula)); return
        small_bgr = cv2.resize(frame, (0, 0), fx=ESCALA_DETEC, fy=ESCALA_DETEC)
        ubs = self._detectar(small_bgr)
        if not ubs:
            self.lbl_login_msg.configure(text="No se detectó rostro.", text_color=C_WARN); self.btn_login_confirmar.configure(state="normal"); return
        small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(small_rgb, [ubs[0]], num_jitters=0, model="small")
        if not encs:
            self.lbl_login_msg.configure(text="No se pudo generar encoding.", text_color=C_WARN); self.btn_login_confirmar.configure(state="normal"); return
        id_u, dist = buscar(encs[0], self._encodings, self._ids)
        if id_u != self._login_usuario["id_usuario"]:
            self.lbl_login_msg.configure(text=f"Rostro no coincide (dist: {dist:.2f}).", text_color=C_ERROR)
            registrar_intento_fallido("Rostro no coincide en login", matricula=matricula, id_usuario=self._login_usuario["id_usuario"]); self.btn_login_confirmar.configure(state="normal"); return
        print(f"[LOGIN] Acceso concedido: {self._login_usuario['nombre']}")
        self._abrir_registro(self._login_usuario)

    def _cancelar_modo(self):
        modo_anterior = self._modo; self._ocultar_overlays(); self._modo = "acceso"; self._en_pausa = False
        self._np_vis = False; self._buffer = []; self._frames_desc = 0
        self._set_estado("escaneando")
        if modo_anterior == "captura":
            self._loop_camara()

    def _build_ov_registro(self):
        self.ov_registro = ctk.CTkFrame(self.frame_video, fg_color="#0F1923", corner_radius=0)
        ctk.CTkLabel(self.ov_registro, text="Registrar nuevo usuario", font=("Helvetica", 15, "bold"), text_color=C_TXT).pack(pady=(24, 2))
        self.lbl_reg_op = ctk.CTkLabel(self.ov_registro, text="", font=("Helvetica", 10), text_color=C_OK)
        self.lbl_reg_op.pack(pady=(0, 12))
        self._entries = {}
        for lbl, key in [("Nombre(s)","nombre"),("Apellido paterno","apellido_p"),("Apellido materno","apellido_m"),("Matrícula","matricula"),("Contraseña","contrasenia")]:
            ctk.CTkLabel(self.ov_registro, text=lbl, font=("Helvetica", 10), text_color=C_TXT2, anchor="w").pack(anchor="w", padx=28)
            e = ctk.CTkEntry(self.ov_registro, width=320, height=38, font=("Helvetica", 12), show="*" if key=="contrasenia" else "")
            e.pack(pady=(0, 4), padx=28)
            e.bind("<FocusIn>", lambda ev, entry=e: self._abrir_teclado(entry))
            self._entries[key] = e
        ctk.CTkLabel(self.ov_registro, text="Rol", font=("Helvetica", 10), text_color=C_TXT2, anchor="w").pack(anchor="w", padx=28)
        self.combo_rol = ctk.CTkComboBox(self.ov_registro, width=320, height=38, font=("Helvetica", 12), values=["ALUMNO","PERSONAL_ESCOLAR"])
        self.combo_rol.pack(pady=(0, 4), padx=28); self.combo_rol.set("ALUMNO")
        self.lbl_reg_err = ctk.CTkLabel(self.ov_registro, text="", font=("Helvetica", 10), text_color=C_ERROR)
        self.lbl_reg_err.pack(pady=2)
        fb = ctk.CTkFrame(self.ov_registro, fg_color="transparent"); fb.pack(pady=10)
        ctk.CTkButton(fb, text="Continuar →", width=150, height=40, fg_color=C_OK, text_color=C_BG, hover_color="#00A88A", font=("Helvetica", 13, "bold"), command=self._reg_continuar).pack(side="left", padx=6)
        ctk.CTkButton(fb, text="Cancelar", width=100, height=40, fg_color="transparent", text_color=C_TXT2, hover_color=C_FRAME, font=("Helvetica", 12), command=self._cancelar_modo).pack(side="left", padx=6)

    def _abrir_registro(self, operador):
        self._modo = "registro"
        mapa = {1:"ADMIN",2:"PERSONAL_AUTORIZADO",3:"PERSONAL_ESCOLAR",4:"ALUMNO"}
        opciones = [mapa[r] for r in roles_asignables(operador["id_rol"]) if r in mapa]
        self.combo_rol.configure(values=opciones); self.combo_rol.set(opciones[-1] if opciones else "ALUMNO")
        self.lbl_reg_op.configure(text=f"Operador: {operador['nombre']} {operador['apellido_p']} ({operador['nombre_rol']})")
        for e in self._entries.values(): e.delete(0, "end")
        self.lbl_reg_err.configure(text="")
        self._ocultar_overlays()
        self.ov_registro.place(relx=0, rely=0, relwidth=1, relheight=1)

    def _reg_continuar(self):
        datos = {k: e.get().strip() for k, e in self._entries.items()}
        mapa  = {"ADMIN":1,"PERSONAL_AUTORIZADO":2,"PERSONAL_ESCOLAR":3,"ALUMNO":4}
        if not all([datos["nombre"], datos["apellido_p"], datos["matricula"], datos["contrasenia"]]):
            self.lbl_reg_err.configure(text="Nombre, apellido, matrícula y contraseña son obligatorios."); return
        if obtener_usuario_por_matricula(datos["matricula"]):
            self.lbl_reg_err.configure(text=f"La matrícula '{datos['matricula']}' ya existe."); return
        datos["id_rol"] = mapa.get(self.combo_rol.get(), 4)
        self._reg_datos = datos; self._abrir_captura()

    def _build_ov_captura(self):
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
        self.lbl_cap_estado = ctk.CTkLabel(self.ov_captura, text="", font=("Helvetica", 12, "bold"), text_color=C_OK)
        self.lbl_cap_estado.pack(pady=2)
        ctk.CTkButton(self.ov_captura, text="Cancelar", fg_color="transparent", text_color=C_TXT2, hover_color=C_FRAME, font=("Helvetica", 12), command=self._cancelar_modo).pack(pady=(10, 0))

    def _abrir_captura(self):
        self._modo = "captura"; self._cap_encodings = []; self._cap_count = 0
        nom = f"{self._reg_datos['nombre']} {self._reg_datos['apellido_p']}"
        self.lbl_cap_nombre.configure(text=nom)
        self.lbl_cap_cnt.configure(text=f"0 / {FOTOS_CAPTURA} fotos")
        self.prog_cap.set(0); self.lbl_cap_estado.configure(text="")
        self._ocultar_overlays()
        self.ov_captura.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._loop_captura()

    def _loop_captura(self):
        if self._modo != "captura" or self._cap_count >= FOTOS_CAPTURA:
            return
        if self._picam:
            frame = self._picam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.flip(frame, 1)
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.after(80, self._loop_captura); return
            frame = cv2.flip(frame, 1)
        try:
            mini = cv2.resize(frame, (320, 240))
            img  = Image.fromarray(cv2.cvtColor(mini, cv2.COLOR_BGR2RGB))
            ci   = ctk.CTkImage(light_image=img, dark_image=img, size=(320, 240))
            self.lbl_cap_video.configure(image=ci, text=""); self._ci_cap = ci
        except Exception:
            pass
        small_bgr = cv2.resize(frame, (0, 0), fx=ESCALA_DETEC, fy=ESCALA_DETEC)
        ubs = self._detectar(small_bgr)
        if ubs:
            small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(small_rgb, [ubs[0]], num_jitters=0, model="small")
            if encs:
                self._cap_encodings.append(encs[0]); self._cap_count += 1
                self.lbl_cap_cnt.configure(text=f"{self._cap_count} / {FOTOS_CAPTURA} fotos")
                self.prog_cap.set(self._cap_count / FOTOS_CAPTURA)
                if self._cap_count >= FOTOS_CAPTURA:
                    self.after(100, self._finalizar_registro); return
        self.after(100, self._loop_captura)

    def _finalizar_registro(self):
        self.lbl_cap_estado.configure(text="Procesando...", text_color=C_WARN); self.update()
        try:
            id_u = registrar_usuario(nombre=self._reg_datos["nombre"], apellido_p=self._reg_datos["apellido_p"], matricula=self._reg_datos["matricula"], contrasenia=self._reg_datos["contrasenia"], id_rol=self._reg_datos["id_rol"], apellido_m=self._reg_datos.get("apellido_m",""))
            enc_prom = np.mean(self._cap_encodings, axis=0)
            guardar_encoding(id_u, json.dumps(enc_prom.tolist()))
            nom_reg = f"{self._reg_datos['nombre']} {self._reg_datos['apellido_p']}"
            print(f"[BD] Registrado: {nom_reg} (ID {id_u})")
            self.lbl_cap_estado.configure(text=f"✓ {nom_reg} registrado. Recargando...", text_color=C_OK)
            def _recargar(n=nom_reg):
                e, ids = cargar_encodings()
                with self._lock:
                    self._encodings = e
                    self._ids = ids
                self.lbl_cap_estado.configure(text=f"✓ {n} registrado", text_color=C_OK)
            import threading as _t; _t.Thread(target=_recargar, daemon=True).start()
            print(f"[BD] Registrado: {nom} (ID {id_u})")
            self.lbl_cap_estado.configure(text=f"✓ {nom} registrado", text_color=C_OK)
            self.after(2000, self._cancelar_modo)
        except Exception as e:
            print(f"[ERROR] {e}"); self.lbl_cap_estado.configure(text=f"Error: {e}", text_color=C_ERROR)

    def _ocultar_overlays(self):
        for ov in [self.ov_numpad, self.ov_login, self.ov_registro, self.ov_captura]:
            ov.place_forget()
        self._ocultar_msg()

    def _set_estado(self, estado):
        self.estado = estado
        cfg = {
            "escaneando" : (C_OK,    "● Escaneando",  ""),
            "verificando": (C_OK,    "● Verificando", "Verificando identidad..."),
            "exito"      : (C_OK,    "✓ Bienvenido/a",""),
            "salida"     : (C_WARN,  "◀ Hasta luego", ""),
            "denegado"   : (C_ERROR, "✗ Denegado",    ""),
        }
        color, badge, inst = cfg.get(estado, (C_TXT2, "●", ""))
        self._set_badge(badge, color)
        if inst: self._set_inst(inst, color)

    def _set_badge(self, t, c): self.lbl_badge.configure(text=t, text_color=c)
    def _set_inst(self, t, c):  self.lbl_inst.configure(text=t, text_color=c)

    def _upd_cnt(self):
        self.lbl_cnt_in.configure(text=f"{self.cnt_in} entrada{'s' if self.cnt_in!=1 else ''}")
        self.lbl_cnt_out.configure(text=f"{self.cnt_out} salida{'s' if self.cnt_out!=1 else ''}")

    def _saludo_hora(self):
        h = datetime.now().hour
        if h < 12: return "Buenos días"
        if h < 19: return "Buenas tardes"
        return "Buenas noches"

    def _pulso(self):
        if self.estado == "escaneando":
            self._pulso_fase = (self._pulso_fase + 1) % 6
            self.lbl_badge.configure(text_color=C_OK if self._pulso_fase < 3 else C_TXT3)
        self._pulso_job = self.after(400, self._pulso)

    def _update_clock(self):
        now = datetime.now()
        self.lbl_hora.configure(text=now.strftime("%H:%M:%S"))
        self.lbl_fecha.configure(text=now.strftime("%d/%m/%Y"))
        h   = now.hour
        self.lbl_saludo.configure(text="Buenos días ☀️" if h<12 else "Buenas tardes 🌤" if h<19 else "Buenas noches 🌙")
        self.after(1000, self._update_clock)

    # ══════════════════════════════════════════════════════════════════════
    #  TECLADO VIRTUAL TÁCTIL
    # ══════════════════════════════════════════════════════════════════════

    def _abrir_teclado(self, entry_target):
        """
        Abre un teclado virtual táctil sobre el widget de video.
        entry_target: el CTkEntry donde se escribirá.
        """
        # Si ya hay un teclado abierto, cerrarlo primero
        if hasattr(self, "_ov_teclado") and self._ov_teclado.winfo_ismapped():
            self._ov_teclado.place_forget()

        self._kb_target  = entry_target
        self._kb_mayus   = False

        # Crear overlay si no existe
        if not hasattr(self, "_ov_teclado"):
            self._ov_teclado = ctk.CTkFrame(
                self, fg_color="#0A1520",
                corner_radius=0)

        # Limpiar contenido anterior
        for w in self._ov_teclado.winfo_children():
            w.destroy()

        self._kb_renderizar()
        # Mostrar en la mitad inferior de la pantalla
        self._ov_teclado.place(relx=0, rely=0.5,
                                relwidth=1, relheight=0.5)
        self._ov_teclado.lift()

    def _kb_renderizar(self):
        """Construye el teclado según estado mayúsculas."""
        for w in self._ov_teclado.winfo_children():
            w.destroy()

        BW, BH = 42, 38
        PAD    = 3

        filas = [
            ["1","2","3","4","5","6","7","8","9","0","⌫"],
            ["Q","W","E","R","T","Y","U","I","O","P"],
            ["A","S","D","F","G","H","J","K","L"],
            ["⇧","Z","X","C","V","B","N","M","-","_"],
        ]

        for r_idx, fila in enumerate(filas):
            for c_idx, tecla in enumerate(fila):
                texto = tecla if not tecla.isalpha() else                         (tecla if self._kb_mayus else tecla.lower())

                if tecla == "⌫":
                    b = ctk.CTkButton(
                        self._ov_teclado, text=tecla,
                        width=BW+6, height=BH,
                        font=("Helvetica", 14),
                        fg_color=C_FRAME, text_color=C_ERROR,
                        hover_color=C_BORDE,
                        border_width=1, border_color=C_BORDE,
                        corner_radius=6,
                        command=self._kb_del)
                elif tecla == "⇧":
                    b = ctk.CTkButton(
                        self._ov_teclado, text=tecla,
                        width=BW+6, height=BH,
                        font=("Helvetica", 14),
                        fg_color=C_OK if self._kb_mayus else C_FRAME,
                        text_color=C_BG if self._kb_mayus else C_TXT,
                        hover_color=C_BORDE,
                        border_width=1, border_color=C_BORDE,
                        corner_radius=6,
                        command=self._kb_toggle_mayus)
                else:
                    b = ctk.CTkButton(
                        self._ov_teclado, text=texto,
                        width=BW, height=BH,
                        font=("Helvetica", 13, "bold"),
                        fg_color=C_FRAME, text_color=C_TXT,
                        hover_color=C_BORDE,
                        border_width=1, border_color=C_BORDE,
                        corner_radius=6,
                        command=lambda t=texto: self._kb_press(t))
                px = 8 if c_idx == 0 else PAD
                b.grid(row=r_idx, column=c_idx, padx=px, pady=PAD)

        # Fila inferior: espacio + cerrar
        fb = ctk.CTkFrame(self._ov_teclado, fg_color="transparent")
        fb.grid(row=4, column=0, columnspan=11,
                padx=8, pady=(2, 4), sticky="ew")

        ctk.CTkButton(fb, text="Espacio", width=200, height=BH,
                       font=("Helvetica", 12),
                       fg_color=C_FRAME, text_color=C_TXT,
                       hover_color=C_BORDE,
                       border_width=1, border_color=C_BORDE,
                       corner_radius=6,
                       command=lambda: self._kb_press(" ")).pack(
                           side="left", padx=4)

        ctk.CTkButton(fb, text="Listo ✓", width=120, height=BH,
                       font=("Helvetica", 12, "bold"),
                       fg_color=C_OK, text_color=C_BG,
                       hover_color="#00A88A",
                       corner_radius=6,
                       command=self._kb_cerrar).pack(side="left", padx=4)

    def _kb_press(self, t):
        if hasattr(self, "_kb_target") and self._kb_target:
            self._kb_target.insert("end", t)

    def _kb_del(self):
        if hasattr(self, "_kb_target") and self._kb_target:
            val = self._kb_target.get()
            self._kb_target.delete(0, "end")
            self._kb_target.insert(0, val[:-1])

    def _kb_toggle_mayus(self):
        self._kb_mayus = not self._kb_mayus
        self._kb_renderizar()

    def _kb_cerrar(self):
        if hasattr(self, "_ov_teclado"):
            self._ov_teclado.place_forget()

    def _cerrar(self):
        if self._pulso_job: self.after_cancel(self._pulso_job)
        if hasattr(self, "_picam") and self._picam:
            self._picam.stop()
        if hasattr(self, "cap") and self.cap: self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = FaceAccess()
    app.mainloop()