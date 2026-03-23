"""
setup_admin.py — Inicialización del primer administrador del sistema.

Solo se necesita correr UNA vez, cuando la BD está vacía.
Si ya existe un admin, el script lo detecta y no hace nada.

Uso:
    python setup_admin.py
"""

import cv2
import json
import sys
import numpy as np
import face_recognition
from db_manager import (
    get_connection,
    guardar_encoding,
    ROL_ADMIN,
)

try:
    from picamera2 import Picamera2
    PICAMERA2_DISPONIBLE = True
except ImportError:
    PICAMERA2_DISPONIBLE = False

MODELO_YUNET  = "face_detection_yunet_2023mar.onnx"
ESCALA_DETEC  = 0.25
FOTOS_OBJETIVO = 10


# ══════════════════════════════════════════════════════════════════════════════
#  VERIFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def ya_existe_admin() -> bool:
    """Retorna True si ya hay al menos un ADMIN activo en la BD."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as total FROM usuarios "
            "WHERE id_rol = ? AND estatus = 1",
            (ROL_ADMIN,)
        ).fetchone()
    return row["total"] > 0


def matricula_existe(matricula: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as total FROM usuarios WHERE matricula = ?",
            (matricula,)
        ).fetchone()
    return row["total"] > 0


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

def crear_detector():
    try:
        det = cv2.FaceDetectorYN.create(
            MODELO_YUNET, "",
            (int(640 * ESCALA_DETEC), int(480 * ESCALA_DETEC)),
            score_threshold=0.80,
            nms_threshold=0.3,
            top_k=1
        )
        print("[INFO] Detector: YuNet")
        return det
    except Exception:
        print("[INFO] Detector: HOG (YuNet no disponible)")
        return None


def detectar(detector, small_bgr, small_rgb):
    if detector:
        h, w = small_bgr.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(small_bgr)
        if faces is None:
            return []
        f = faces[0]
        if float(f[14]) < 0.80:
            return []
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


# ══════════════════════════════════════════════════════════════════════════════
#  CAPTURA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def pedir_datos() -> dict:
    print("\n" + "="*50)
    print("  DATOS DEL ADMINISTRADOR")
    print("="*50)

    while True:
        nombre = input("Nombre(s)         : ").strip()
        if nombre:
            break
        print("[!] El nombre no puede estar vacío.")

    while True:
        apellido_p = input("Apellido paterno  : ").strip()
        if apellido_p:
            break
        print("[!] El apellido no puede estar vacío.")

    apellido_m = input("Apellido materno  : ").strip()

    while True:
        matricula = input("Matrícula         : ").strip()
        if not matricula:
            print("[!] La matrícula no puede estar vacía.")
            continue
        if matricula_existe(matricula):
            print(f"[!] La matrícula '{matricula}' ya existe en la BD.")
            continue
        break

    while True:
        contrasenia = input("Contraseña        : ").strip()
        if len(contrasenia) < 4:
            print("[!] La contraseña debe tener al menos 4 caracteres.")
            continue
        confirmacion = input("Confirmar contraseña: ").strip()
        if contrasenia != confirmacion:
            print("[!] Las contraseñas no coinciden.")
            continue
        break

    return {
        "nombre"     : nombre,
        "apellido_p" : apellido_p,
        "apellido_m" : apellido_m,
        "matricula"  : matricula,
        "contrasenia": contrasenia,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CAPTURA DE ROSTRO
# ══════════════════════════════════════════════════════════════════════════════

def capturar_encoding(nombre_completo: str) -> list | None:
    """Abre la cámara y captura FOTOS_OBJETIVO encodings del admin."""
    print(f"\n[INFO] Capturando rostro de: {nombre_completo}")
    print(f"[INFO] Se necesitan {FOTOS_OBJETIVO} capturas.")
    print("[INFO] Mira de frente a la cámara.")
    print("[INFO] Presiona ESC para cancelar.\n")

    detector = crear_detector()

    if PICAMERA2_DISPONIBLE:
        picam = Picamera2()
        config = picam.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"})
        picam.configure(config)
        picam.start()
        cap = None
        print("[CAMARA] Usando picamera2 (CSI)")
    else:
        picam = None
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] No se pudo abrir la cámara.")
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[CAMARA] Usando OpenCV (USB/webcam)")

    encodings = []

    while True:
        if picam:
            frame = picam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.flip(frame, 1)
        else:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
        small_bgr = cv2.resize(frame, (0, 0),
                                fx=ESCALA_DETEC, fy=ESCALA_DETEC)
        small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        ubs       = detectar(detector, small_bgr, small_rgb)

        for (top, right, bottom, left) in ubs:
            esc = int(1 / ESCALA_DETEC)
            cv2.rectangle(frame,
                          (left*esc, top*esc),
                          (right*esc, bottom*esc),
                          (0, 212, 170), 2)

            encs = face_recognition.face_encodings(
                small_rgb, [(top, right, bottom, left)],
                num_jitters=0, model="small")
            if encs:
                encodings.append(encs[0])

        # Progreso
        actual = len(encodings)
        bw     = int((actual / FOTOS_OBJETIVO) * 300)
        h_f    = frame.shape[0]
        cv2.rectangle(frame, (10, h_f-30), (310, h_f-10), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, h_f-30), (10+bw, h_f-10), (0, 212, 170), -1)
        cv2.putText(frame, f"{actual}/{FOTOS_OBJETIVO} capturas",
                    (10, h_f-35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Admin: {nombre_completo}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 170), 2)

        if actual >= FOTOS_OBJETIVO:
            cv2.putText(frame, "Listo!",
                        (frame.shape[1]//2 - 50, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 212, 170), 3)

        cv2.imshow("Setup Admin — ESC para cancelar", frame)
        key = cv2.waitKey(1)

        if key == 27:
            print("\n[CANCELADO] Setup cancelado por el usuario.")
            cap.release()
            cv2.destroyAllWindows()
            return None

        if actual >= FOTOS_OBJETIVO:
            break

    if picam:
        picam.stop()
    if cap:
        cap.release()
    cv2.destroyAllWindows()

    if not encodings:
        print("[ERROR] No se capturó ningún encoding.")
        return None

    return encodings


# ══════════════════════════════════════════════════════════════════════════════
#  GUARDAR EN BD
# ══════════════════════════════════════════════════════════════════════════════

def crear_admin(datos: dict, encodings: list) -> int:
    """Inserta el admin en la BD y guarda su encoding."""
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO usuarios "
            "(nombre, apellido_p, apellido_m, matricula, contrasenia, id_rol) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                datos["nombre"],
                datos["apellido_p"],
                datos["apellido_m"],
                datos["matricula"],
                datos["contrasenia"],
                ROL_ADMIN,
            )
        )
        id_usuario = cursor.lastrowid

    enc_promedio = np.mean(encodings, axis=0)
    guardar_encoding(id_usuario, json.dumps(enc_promedio.tolist()))

    return id_usuario


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*50)
    print("  FACEACCESS — SETUP INICIAL")
    print("="*50)

    # Verificar si ya hay admin
    if ya_existe_admin():
        print("\n[OK] Ya existe un administrador en la BD.")
        print("[INFO] No es necesario correr este script de nuevo.")
        print("[INFO] Usa ReconocimientoFacial.py directamente.")
        sys.exit(0)

    print("\n[INFO] No se encontró ningún administrador.")
    print("[INFO] Vamos a crear el primer administrador del sistema.\n")

    # Pedir datos
    datos = pedir_datos()

    nombre_completo = f"{datos['nombre']} {datos['apellido_p']}"

    # Confirmar
    print("\n" + "="*50)
    print("  CONFIRMAR DATOS")
    print("="*50)
    print(f"  Nombre    : {nombre_completo}")
    print(f"  Matrícula : {datos['matricula']}")
    print(f"  Rol       : ADMIN")
    print("="*50)

    confirmar = input("\n¿Los datos son correctos? (s/n): ").strip().lower()
    if confirmar != "s":
        print("[CANCELADO] Vuelve a correr el script para intentarlo de nuevo.")
        sys.exit(0)

    # Capturar rostro
    encodings = capturar_encoding(nombre_completo)
    if encodings is None:
        print("[ERROR] No se pudo capturar el rostro. Intenta de nuevo.")
        sys.exit(1)

    # Guardar en BD
    try:
        id_u = crear_admin(datos, encodings)
        print("\n" + "="*50)
        print("  ✓ ADMINISTRADOR CREADO EXITOSAMENTE")
        print("="*50)
        print(f"  Nombre    : {nombre_completo}")
        print(f"  Matrícula : {datos['matricula']}")
        print(f"  ID        : {id_u}")
        print(f"  Encodings : {len(encodings)} capturas")
        print("="*50)
        print("\n[LISTO] Ahora puedes correr ReconocimientoFacial.py")
    except Exception as e:
        print(f"\n[ERROR] No se pudo guardar en la BD: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()