"""
core/captura.py — Captura de rostros en escala de grises y almacenamiento
"""

import cv2
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import (
    get_connection,
    guardar_encoding,
    DATA_DIR,
)
from models.entrenadoRF import entrenar, FACE_SIZE
from ui.constantes import HAAR_CASCADE, ESCALA_DETEC, MIN_VECINOS, MIN_TAMANO_RELAT

FOTOS_OBJETIVO = 30


def crear_detector():
    """Crear detector de rostros con Haar Cascade."""
    detector = cv2.CascadeClassifier(HAAR_CASCADE)
    if detector.empty():
        raise RuntimeError(f"No se encontró: {HAAR_CASCADE}")
    print(f"[INFO] Haar Cascade cargado.")
    return detector


def detectar_rostro(detector, frame_gray_small):
    """
    Detectar el rostro más prominente.
    Retorna (x, y, w, h) del rostro más grande, o None.
    """
    h_sm, w_sm = frame_gray_small.shape[:2]
    min_size = (
        max(int(w_sm * MIN_TAMANO_RELAT), 20),
        max(int(h_sm * MIN_TAMANO_RELAT), 20)
    )
    rostros = detector.detectMultiScale(
        frame_gray_small,
        scaleFactor=1.1,
        minNeighbors=MIN_VECINOS,
        minSize=min_size
    )
    if len(rostros) == 0:
        return None
    return max(rostros, key=lambda r: r[2] * r[3])


def seleccionar_usuario():
    """Solicitar matrícula y retornar datos de usuario."""
    matricula = input("Matrícula del usuario: ").strip()
    with get_connection() as conn:
        usuario = conn.execute(
            "SELECT * FROM usuarios WHERE matricula = ? AND estatus = 1",
            (matricula.upper(),)
        ).fetchone()
    if not usuario:
        print(f"[ERROR] No existe usuario con matrícula '{matricula}'.")
        return None
    print(f"[OK] {usuario['nombre']} {usuario['apellido_p']} (ID {usuario['id_usuario']})")
    return usuario


def capturar(usuario):
    """
    Captura FOTOS_OBJETIVO imágenes del rostro del usuario.
    Guarda cada imagen en escala de grises (200x200) en la carpeta del usuario.
    Retorna True si tuvo éxito, False si fue cancelada o tuvo error.
    """
    id_usuario = usuario["id_usuario"]
    nombre = f"{usuario['nombre']}_{usuario['apellido_p']}"
    detector = crear_detector()

    # Crear carpeta para este usuario
    carpeta = os.path.join(DATA_DIR, f"{id_usuario}_{nombre}")
    os.makedirs(carpeta, exist_ok=True)
    print(f"[INFO] Guardando imágenes en: {carpeta}")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return False

    capturadas = 0
    print(f"[INFO] Capturando {FOTOS_OBJETIVO} imágenes del rostro.")
    print("[INFO] Mira de frente a la cámara. Mueve ligeramente la cabeza.")
    print("[INFO] Presiona ESC para cancelar.")

    while capturadas < FOTOS_OBJETIVO:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Reducir tamaño para detección más rápida
        small = cv2.resize(gray, (0, 0), fx=ESCALA_DETEC, fy=ESCALA_DETEC)

        rostro_coords = detectar_rostro(detector, small)

        if rostro_coords is not None:
            # Escalar coordenadas de vuelta al frame original
            esc = int(1 / ESCALA_DETEC)
            x, y, w, h = rostro_coords
            x, y, w, h = x * esc, y * esc, w * esc, h * esc

            # Recortar el rostro del frame en escala de grises
            rostro_crop = gray[y : y + h, x : x + w]

            # Normalizar tamaño
            rostro_res = cv2.resize(rostro_crop, FACE_SIZE)

            # Guardar imagen en disco
            ruta_img = os.path.join(carpeta, f"rostro_{capturadas:03d}.jpg")
            cv2.imwrite(ruta_img, rostro_res)
            capturadas += 1

            # Dibujar rectángulo verde
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Barra de progreso
        bw = int((capturadas / FOTOS_OBJETIVO) * 300)
        cv2.rectangle(frame, (10, frame.shape[0] - 30), (310, frame.shape[0] - 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, frame.shape[0] - 30), (10 + bw, frame.shape[0] - 10), (0, 212, 170), -1)
        cv2.putText(
            frame,
            f"{capturadas}/{FOTOS_OBJETIVO} capturas",
            (10, frame.shape[0] - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            frame,
            f"Usuario: {usuario['nombre']} {usuario['apellido_p']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 212, 170),
            2,
        )

        if capturadas >= FOTOS_OBJETIVO:
            cv2.putText(
                frame,
                "Listo!",
                (frame.shape[1] // 2 - 50, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 212, 170),
                3,
            )

        cv2.imshow("Captura de rostro — ESC para cancelar", frame)

        key = cv2.waitKey(1)
        if key == 27:
            print("\n[CANCELADO] Captura cancelada.")
            cap.release()
            cv2.destroyAllWindows()
            return False

    cap.release()
    cv2.destroyAllWindows()

    print(f"[OK] {capturadas} imágenes guardadas en: {carpeta}")

    # Guardar la ruta de la carpeta en la BD
    guardar_encoding(id_usuario, carpeta)

    # Re-entrenar el modelo LBPH
    print("[INFO] Re-entrenando modelo LBPH...")
    ok = entrenar()
    if ok:
        print("[OK] Modelo LBPH actualizado.")
    else:
        print("[AVISO] No se pudo re-entrenar.")

    return True


if __name__ == "__main__":
    u = seleccionar_usuario()
    if u:
        capturar(u)
