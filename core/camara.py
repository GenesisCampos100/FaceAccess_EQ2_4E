"""
core/camara.py
Abstracción de cámara: Picamera2 (CSI) o webcam USB.
FaceAccess usa self._camara.leer() sin saber qué cámara está conectada.
"""

import cv2

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
        self.cap    = None
        self._lock  = None   # se asigna desde FaceAccess para compartir el mismo lock

    def iniciar(self, lock=None):
        """
        Inicializa la cámara disponible.
        lock — threading.Lock() compartido con FaceAccess (opcional)
        """
        self._lock = lock

        if PICAMERA2_DISPONIBLE:
            self._picam = Picamera2()
            config = self._picam.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"})
            self._picam.configure(config)
            self._picam.start()
            print("[CAMARA] Usando picamera2 (CSI)")
        else:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print("[CAMARA] Usando OpenCV (USB/webcam)")

    def leer(self):
        """
        Lee un frame de la cámara activa.
        Retorna frame BGR con flip horizontal aplicado, o None si falla.
        """
        if self._picam:
            if self._lock:
                with self._lock:
                    frame = self._picam.capture_array()
            else:
                frame = self._picam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return cv2.flip(frame, 1)

        if self.cap:
            if self._lock:
                with self._lock:
                    ret, frame = self.cap.read()
            else:
                ret, frame = self.cap.read()
            if not ret:
                return None
            return cv2.flip(frame, 1)

        return None

    def liberar(self):
        """Libera los recursos de la cámara al cerrar la app."""
        if self._picam:
            self._picam.stop()
        if self.cap:
            self.cap.release()

    @property
    def usa_picam(self):
        return self._picam is not None
