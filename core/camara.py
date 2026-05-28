"""
core/camara.py
Abstracción de cámara: Picamera2 (CSI) o webcam USB.
FaceAccess usa self._camara.leer() sin saber qué cámara está conectada.
"""

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    PICAMERA2_DISPONIBLE = True
except ImportError:
    PICAMERA2_DISPONIBLE = False


class CamaraManager:
    """
    Gestiona la inicialización y lectura de frames de la cámara.

    Uso desde FaceAccess:
        self._camara = CamaraManager()
        self._camara.iniciar()
        frame = self._camara.leer()   # BGR, ya con flip horizontal
        self._camara.liberar()
    """

    def __init__(self):
        self._picam = None
        self.cap = None
        self._lock = None

        gamma = 1.15
        self._gamma_lut = np.array([
            np.clip((i / 255.0) ** (1.0 / gamma) * 255.0, 0, 255).astype(np.uint8)
            for i in range(256)
        ])
        self._clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    def iniciar(self, lock=None):
        """
        Inicializa la cámara disponible.
        lock — threading.Lock() compartido con FaceAccess (opcional)
        """
        self._lock = lock

        if PICAMERA2_DISPONIBLE:
            try:
                self._picam = Picamera2()
                config = self._picam.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                self._picam.configure(config)
                self._picam.start()
                print("[CAMARA] Usando picamera2 (CSI)")
                return
            except Exception as exc:
                print(f"[CAMARA] Picamera2 no pudo inicializarse, se usa webcam: {exc}")
                self._picam = None

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        print("[CAMARA] Usando OpenCV (USB/webcam)")

    def _procesar_colorimetria(self, frame):
        """
        Aplica mejoras visuales al frame sin alterar sus dimensiones ni formato.
        Optimizado para reconocimiento facial y despliegue en UI.
        """
        if frame is None:
            return None

        frame = cv2.LUT(frame, self._gamma_lut)

        b, g, r = cv2.split(frame)
        b_avg, g_avg, r_avg = np.mean(b), np.mean(g), np.mean(r)
        if b_avg > 0 and g_avg > 0 and r_avg > 0:
            k = (b_avg + g_avg + r_avg) / 3.0
            kb, kg, kr = k / b_avg, k / g_avg, k / r_avg
            b = np.clip(b * kb, 0, 255).astype(np.uint8)
            g = np.clip(g * kg, 0, 255).astype(np.uint8)
            r = np.clip(r * kr, 0, 255).astype(np.uint8)
            frame = cv2.merge((b, g, r))

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, _b = cv2.split(lab)
        l_equ = self._clahe.apply(l)
        lab = cv2.merge((l_equ, a, _b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return frame

    def leer(self):
        """
        Lee un frame de la cámara activa.
        Retorna frame BGR con flip horizontal y mejoras colorimétricas aplicadas, o None si falla.
        """
        frame_crudo = None

        if self._picam:
            if self._lock:
                with self._lock:
                    frame_crudo = self._picam.capture_array()
            else:
                frame_crudo = self._picam.capture_array()
            if frame_crudo is not None:
                frame_crudo = cv2.cvtColor(frame_crudo, cv2.COLOR_RGB2BGR)

        elif self.cap:
            if self._lock:
                with self._lock:
                    ret, frame_crudo = self.cap.read()
            else:
                ret, frame_crudo = self.cap.read()
            if not ret:
                frame_crudo = None

        if frame_crudo is not None:
            frame_mejorado = self._procesar_colorimetria(frame_crudo)
            return cv2.flip(frame_mejorado, 1)

        return None

    def liberar(self):
        """Libera los recursos de la cámara al cerrar la app."""
        if self._picam:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception as e:
                print(f"[CAMARA] Error al liberar picamera2: {e}")
            finally:
                self._picam = None
        if self.cap:
            self.cap.release()
            self.cap = None

    @property
    def usa_picam(self):
        return self._picam is not None
