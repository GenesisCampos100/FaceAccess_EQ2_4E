"""
capturar_rostro.py — Captura imágenes del rostro y las guarda en disco.

ARQUITECTURA (corregida):
  - Las imágenes se guardan en disco: data_rostros/<id>_<nombre>/rostro_000.jpg
  - La BD solo recibe el metadato (ruta de la carpeta) via guardar_encoding().
  - La tabla imagenes_biometricas (BLOB) ya no se usa.
"""

import cv2
import os
import numpy as np
from db_manager import (
    get_connection,
    guardar_encoding,
    DATA_DIR,
)
from entrenadoRF import entrenar, FACE_SIZE

# ─── Parámetros de captura ────────────────────────────────────────────────────
FOTOS_OBJETIVO = 30
HAAR_CASCADE   = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
ESCALA_DETEC   = 0.5
MIN_VECINOS    = 5
MIN_TAMANO     = (60, 60)


def crear_detector():
    detector = cv2.CascadeClassifier(HAAR_CASCADE)
    if detector.empty():
        raise RuntimeError(f"No se encontró el archivo: {HAAR_CASCADE}")
    print("[INFO] Haar Cascade cargado.")
    return detector


def detectar_rostro(detector, frame_gray_small):
    rostros = detector.detectMultiScale(
        frame_gray_small,
        scaleFactor=1.1,
        minNeighbors=MIN_VECINOS,
        minSize=MIN_TAMANO
    )
    if len(rostros) == 0:
        return None
    return max(rostros, key=lambda r: r[2] * r[3])


def seleccionar_usuario():
    matricula = input("Matrícula del usuario: ").strip()
    with get_connection() as conn:
        usuario = conn.execute(
            "SELECT * FROM usuarios WHERE matricula = ? AND estatus = 1",
            (matricula,)
        ).fetchone()
    if not usuario:
        print(f"[ERROR] No existe usuario con matrícula '{matricula}'.")
        return None
    print(f"[OK] {usuario['nombre']} {usuario['apellido_p']} (ID {usuario['id_usuario']})")
    return usuario


def capturar(usuario):
    """
    Abre la cámara, captura FOTOS_OBJETIVO imágenes en escala de grises
    y las guarda en disco en data_rostros/<id>_<nombre>/.
    Luego registra el metadato en la BD y re-entrena el modelo LBPH.
    """
    id_usuario = usuario["id_usuario"]
    nombre     = usuario["nombre"]
    apellido   = usuario["apellido_p"]

    # ── Preparar carpeta en disco ─────────────────────────────────────────────
    nom_carpeta = f"{id_usuario}_{nombre}_{apellido}"
    carpeta     = os.path.join(DATA_DIR, nom_carpeta)
    os.makedirs(carpeta, exist_ok=True)

    detector = crear_detector()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return False

    imagenes   = []
    capturadas = 0

    print(f"[INFO] Capturando {FOTOS_OBJETIVO} imágenes del rostro.")
    print("[INFO] Mira de frente a la cámara. Mueve ligeramente la cabeza.")
    print("[INFO] Presiona ESC para cancelar.")

    while capturadas < FOTOS_OBJETIVO:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (0, 0), fx=ESCALA_DETEC, fy=ESCALA_DETEC)

        rostro_coords = detectar_rostro(detector, small)

        if rostro_coords is not None:
            esc = int(1 / ESCALA_DETEC)
            x, y, w, h = rostro_coords
            x, y, w, h = x*esc, y*esc, w*esc, h*esc

            rostro_crop = gray[y:y+h, x:x+w]
            rostro_res  = cv2.resize(rostro_crop, FACE_SIZE)

            imagenes.append(rostro_res)
            capturadas += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        bw = int((capturadas / FOTOS_OBJETIVO) * 300)
        cv2.rectangle(frame, (10, frame.shape[0]-30),
                      (310, frame.shape[0]-10), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, frame.shape[0]-30),
                      (10+bw, frame.shape[0]-10), (0, 212, 170), -1)
        cv2.putText(frame, f"{capturadas}/{FOTOS_OBJETIVO} capturas",
                    (10, frame.shape[0]-35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame,
                    f"Usuario: {nombre} {apellido}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 170), 2)

        if capturadas >= FOTOS_OBJETIVO:
            cv2.putText(frame, "Listo!",
                        (frame.shape[1]//2 - 50, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 212, 170), 3)

        cv2.imshow("Captura de rostro — ESC para cancelar", frame)

        key = cv2.waitKey(1)
        if key == 27:
            print("\n[CANCELADO] Captura cancelada.")
            cap.release()
            cv2.destroyAllWindows()
            return False

    cap.release()
    cv2.destroyAllWindows()

    # ── Guardar imágenes en disco ─────────────────────────────────────────────
    print(f"[INFO] Guardando {len(imagenes)} imágenes en: {carpeta}")
    for idx, img_gray in enumerate(imagenes):
        ruta_img = os.path.join(carpeta, f"rostro_{idx:03d}.jpg")
        cv2.imwrite(ruta_img, img_gray)

    # ── Registrar metadato en BD ──────────────────────────────────────────────
    guardar_encoding(id_usuario, carpeta)

    # ── Re-entrenar modelo LBPH ───────────────────────────────────────────────
    print("[INFO] Re-entrenando modelo LBPH...")
    ok = entrenar()
    if ok:
        print("[OK] Modelo LBPH actualizado.")
    else:
        print("[AVISO] No se pudo re-entrenar. Ejecuta entrenadoRF.py manualmente.")

    return True


if __name__ == "__main__":
    u = seleccionar_usuario()
    if u:
        capturar(u)