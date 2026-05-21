
"""
core/reconocimiento.py
Funciones puras de reconocimiento facial con LBPH.
No importa nada de UI ni de db_manager.
"""

import cv2
from models.entrenadoRF import cargar_modelo_lbph, FACE_SIZE
from ui.constantes import UMBRAL_CONFIANZA


def cargar_encodings():
    """
    Carga el modelo LBPH desde disco.
    Retorna (recognizer, []) — lista vacía por compatibilidad con el resto del código.
    """
    recognizer = cargar_modelo_lbph()
    print(f"[INFO] Modelo LBPH {'listo' if recognizer else 'NO encontrado'}.")
    return recognizer, []


def buscar(rostro_gray, recognizer):
    """
    Reconoce un rostro usando LBPH.
    Recibe el crop CRUDO en escala de grises — aplica preprocesamiento internamente.

    El preprocesamiento (CLAHE + bilateralFilter + normalize) debe ser
    IDÉNTICO al aplicado durante el entrenamiento en entrenadoRF.py.

    Parámetros:
        rostro_gray  — crop crudo del rostro en escala de grises (numpy array)
        recognizer   — modelo LBPH cargado (cv2.face.LBPHFaceRecognizer)

    Retorna:
        (id_usuario, confianza) si confianza < UMBRAL_CONFIANZA
        (None, confianza)       si es desconocido
    """
    if recognizer is None:
        return None, 999.0

    rostro_res = cv2.resize(rostro_gray, FACE_SIZE)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    rostro_res = clahe.apply(rostro_res)
    rostro_res = cv2.bilateralFilter(rostro_res, 5, 75, 75)
    rostro_res = cv2.normalize(rostro_res, None, 0, 255, cv2.NORM_MINMAX)

    label, confianza = recognizer.predict(rostro_res)

    if confianza < UMBRAL_CONFIANZA:
        return label, confianza
    else:
        return None, confianza
