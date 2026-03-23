"""
capturar_rostro.py — Captura encodings del rostro y los guarda en BD.

- NO guarda fotos en disco
- Solo guarda el vector de 128 puntos en datos_biometricos
- Usa YuNet para detección (mismo que ReconocimientoFacial.py)
- Valida que el rostro esté de frente antes de aceptar el encoding
"""

import cv2
import json
import numpy as np
import face_recognition
from db_manager import get_connection, guardar_encoding

FOTOS_OBJETIVO    = 10
MODELO_YUNET      = "face_detection_yunet_2023mar.onnx"
ESCALA_DETEC      = 0.25
SCORE_MIN         = 0.85


# ── Detector (mismo que en ReconocimientoFacial.py) ───────────────────────────

def crear_detector():
    try:
        det = cv2.FaceDetectorYN.create(
            MODELO_YUNET, "",
            (int(640 * ESCALA_DETEC), int(480 * ESCALA_DETEC)),
            score_threshold=SCORE_MIN,
            nms_threshold=0.3,
            top_k=1
        )
        print("[INFO] Usando YuNet.")
        return det
    except Exception:
        print("[INFO] YuNet no disponible, usando HOG.")
        return None


def detectar(detector, small_bgr, small_rgb):
    """Misma lógica de validación que _detectar() en ReconocimientoFacial.py."""
    if detector:
        h, w = small_bgr.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(small_bgr)
        if faces is None:
            return []
        f = faces[0]

        # Score mínimo
        if float(f[14]) < SCORE_MIN:
            return []

        # Nariz entre los ojos (valida frente)
        if len(f) >= 14:
            ojo_der_x = float(f[4])
            ojo_izq_x = float(f[6])
            nariz_x   = float(f[8])
            x_min = min(ojo_der_x, ojo_izq_x)
            x_max = max(ojo_der_x, ojo_izq_x)
            margen = (x_max - x_min) * 0.25
            if not (x_min - margen < nariz_x < x_max + margen):
                return []
            if abs(ojo_izq_x - ojo_der_x) < 8:
                return []

        return [(int(f[1]), int(f[0]+f[2]), int(f[1]+f[3]), int(f[0]))]
    else:
        return face_recognition.face_locations(
            small_rgb, model="hog", number_of_times_to_upsample=0)


# ── Selección de usuario ──────────────────────────────────────────────────────

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


# ── Captura ───────────────────────────────────────────────────────────────────

def capturar(usuario):
    id_usuario = usuario["id_usuario"]
    detector   = crear_detector()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    encodings = []
    print(f"[INFO] Capturando {FOTOS_OBJETIVO} encodings.")
    print("[INFO] Mira de frente a la cámara. Mueve ligeramente la cabeza.")
    print("[INFO] Presiona ESC para cancelar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        small_bgr = cv2.resize(frame, (0, 0),
                                fx=ESCALA_DETEC, fy=ESCALA_DETEC)
        small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        ubs       = detectar(detector, small_bgr, small_rgb)

        for (top, right, bottom, left) in ubs:
            # Escalar coords para dibujar en frame original
            esc = int(1 / ESCALA_DETEC)
            cv2.rectangle(frame,
                          (left*esc, top*esc),
                          (right*esc, bottom*esc),
                          (0, 255, 0), 2)

            encs = face_recognition.face_encodings(
                small_rgb, [(top, right, bottom, left)],
                num_jitters=0, model="small")
            if encs:
                encodings.append(encs[0])

        # Barra de progreso visual
        total  = FOTOS_OBJETIVO
        actual = len(encodings)
        bw     = int((actual / total) * 300)
        cv2.rectangle(frame, (10, frame.shape[0]-30),
                      (310, frame.shape[0]-10), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, frame.shape[0]-30),
                      (10+bw, frame.shape[0]-10), (0, 212, 170), -1)
        cv2.putText(frame, f"{actual}/{total} encodings",
                    (10, frame.shape[0]-35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if actual >= total:
            cv2.putText(frame, "Listo!",
                        (frame.shape[1]//2 - 40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 212, 170), 2)

        cv2.imshow("Captura de rostro — ESC para cancelar", frame)

        if cv2.waitKey(1) == 27 or actual >= total:
            break

    cap.release()
    cv2.destroyAllWindows()

    if not encodings:
        print("[ERROR] No se capturó ningún encoding.")
        return

    enc_promedio = np.mean(encodings, axis=0)
    guardar_encoding(id_usuario, json.dumps(enc_promedio.tolist()))
    print(f"[OK] Encoding guardado para ID {id_usuario} ({len(encodings)} capturas).")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    u = seleccionar_usuario()
    if u:
        capturar(u)