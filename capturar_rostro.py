"""
capturar_rostro.py — Captura imágenes del rostro y genera el encoding
con face_recognition.

Diferencia vs LBPH:
  · Solo necesitas 5-10 fotos buenas (no 100)
  · El encoding (128 puntos) se genera aquí mismo y se guarda en la BD
  · No necesitas ejecutar entrenadoRF.py después — este script hace todo
  · La carpeta de fotos se guarda igual como respaldo

Flujo:
  1. Pide matrícula del usuario (debe existir en BD)
  2. Abre la cámara y captura FOTOS_OBJETIVO fotos del rostro
  3. Genera el encoding promedio de todas las fotos
  4. Guarda el encoding en datos_biometricos
"""

import cv2
import os
import json
import imutils
import face_recognition
import numpy as np
from db_manager import (
    get_connection,
    guardar_encoding,
)

# ─── Configuración ────────────────────────────────────────────────────────────
DATA_PATH     = "C:/xampp/htdocs/FaceAccess_EQ2_4E/data"
FOTOS_OBJETIVO = 10    # con face_recognition 10 fotos son más que suficientes


# ─── Selección del usuario ────────────────────────────────────────────────────
def seleccionar_usuario():
    matricula = input("Ingresa la matrícula del usuario a capturar: ").strip()
    with get_connection() as conn:
        usuario = conn.execute(
            "SELECT * FROM usuarios WHERE matricula = ? AND estatus = 1",
            (matricula,)
        ).fetchone()
    if not usuario:
        print(f"[ERROR] No se encontró usuario activo con matrícula '{matricula}'.")
        return None
    nombre_completo = f"{usuario['nombre']} {usuario['apellido_p']}"
    print(f"[OK] Usuario encontrado: {nombre_completo} (ID {usuario['id_usuario']})")
    return usuario


# ─── Captura y generación de encoding ────────────────────────────────────────
def capturar(usuario):
    id_usuario  = usuario["id_usuario"]
    nombre      = f"{usuario['nombre']}_{usuario['apellido_p']}"
    folder_name = f"{id_usuario}_{nombre}"
    person_path = os.path.join(DATA_PATH, folder_name)

    if not os.path.exists(person_path):
        os.makedirs(person_path)
        print(f"[INFO] Carpeta creada: {person_path}")

    cap          = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    encodings_capturados = []   # lista de vectores de 128 puntos
    count                = 0

    print(f"[INFO] Capturando {FOTOS_OBJETIVO} fotos.")
    print("[INFO] Mueve ligeramente la cabeza en diferentes ángulos.")
    print("[INFO] Presiona ESC para cancelar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame    = imutils.resize(frame, width=640)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # face_recognition usa RGB

        # Detectar ubicaciones de rostros en el frame
        ubicaciones = face_recognition.face_locations(rgb, model="hog")

        for (top, right, bottom, left) in ubicaciones:
            # Dibujar recuadro
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Generar encoding del rostro detectado
            encs = face_recognition.face_encodings(rgb, [(top, right, bottom, left)])
            if encs:
                encodings_capturados.append(encs[0])
                # Guardar foto como respaldo
                rostro = frame[top:bottom, left:right]
                rostro = cv2.resize(rostro, (150, 150))
                cv2.imwrite(os.path.join(person_path, f"rostro_{count}.jpg"), rostro)
                count += 1

        # Progreso en pantalla
        cv2.putText(frame, f"Fotos: {count}/{FOTOS_OBJETIVO}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

        if count >= FOTOS_OBJETIVO:
            cv2.putText(frame, "Listo! Procesando...",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Captura de rostro", frame)

        k = cv2.waitKey(1)
        if k == 27 or count >= FOTOS_OBJETIVO:
            break

    cap.release()
    cv2.destroyAllWindows()

    if not encodings_capturados:
        print("[ERROR] No se capturó ningún rostro. Inténtalo de nuevo.")
        return

    # ── Generar encoding promedio de todas las capturas ───────────────────
    # Promediamos todos los vectores para tener una representación más robusta
    encoding_promedio = np.mean(encodings_capturados, axis=0)
    encoding_str      = json.dumps(encoding_promedio.tolist())

    # Guardar en BD
    guardar_encoding(id_usuario, encoding_str)

    nombre_completo = f"{usuario['nombre']} {usuario['apellido_p']}"
    print(f"[OK] Encoding generado con {len(encodings_capturados)} fotos.")
    print(f"[BD] Encoding guardado para: {nombre_completo} (ID {id_usuario})")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    usuario = seleccionar_usuario()
    if usuario:
        capturar(usuario)
