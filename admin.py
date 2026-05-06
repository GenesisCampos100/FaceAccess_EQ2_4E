"""
admin.py — Inicialización del primer administrador del sistema.

Solo se necesita correr UNA vez, cuando la BD está vacía.
Si ya existe un admin, el script lo detecta y no hace nada.
"""

import cv2
import os
import sys
import numpy as np
from db_manager import (
    get_connection,
    guardar_encoding,
    DATA_DIR,
    ROL_ADMIN,
)
from entrenadoRF import entrenar, FACE_SIZE

try:
    from picamera2 import Picamera2
    PICAMERA2_DISPONIBLE = True
except ImportError:
    PICAMERA2_DISPONIBLE = False

# ─── Parámetros ───────────────────────────────────────────────────────────────
HAAR_CASCADE   = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FOTOS_OBJETIVO = 30       # más fotos = modelo más robusto
ESCALA_DETEC   = 0.5      # factor de reducción para detección más rápida
MIN_VECINOS    = 5
MIN_TAMANO     = (60, 60)


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
#  DETECTOR HAAR CASCADE
# ══════════════════════════════════════════════════════════════════════════════

def crear_detector():
    """
    Crea el detector de rostros con Haar Cascade.
    Es un método clásico, ni redes neuronales.
    """
    detector = cv2.CascadeClassifier(HAAR_CASCADE)
    if detector.empty():
        raise RuntimeError(f"No se pudo cargar Haar Cascade: {HAAR_CASCADE}")
    print(f"[INFO] Detector: Haar Cascade")
    return detector


def detectar_rostro(detector, gray_small):
    """
    Detecta el rostro principal en una imagen en escala de grises.
    Retorna (x, y, w, h) del rostro más grande, o None si no hay detección.

    scaleFactor=1.1  → busca rostros a múltiples escalas
    minNeighbors=5   → cantidad de detecciones vecinas requeridas
    minSize          → descarta objetos pequeños
    """
    rostros = detector.detectMultiScale(
        gray_small,
        scaleFactor=1.1,
        minNeighbors=MIN_VECINOS,
        minSize=MIN_TAMANO
    )
    if len(rostros) == 0:
        return None
    return max(rostros, key=lambda r: r[2] * r[3])


# ══════════════════════════════════════════════════════════════════════════════
#  CAPTURA DE DATOS DEL ADMINISTRADOR
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
#  CAPTURA DE IMÁGENES DEL ROSTRO
# ══════════════════════════════════════════════════════════════════════════════

def capturar_rostro(id_usuario: int, nombre_completo: str) -> str | None:
    """
    Abre la cámara y captura FOTOS_OBJETIVO imágenes del rostro en escala de grises.
    Las guarda en DATA_DIR/<id>_<nombre>/.
    Retorna la ruta de la carpeta, o None si se canceló.
    """
    print(f"\n[INFO] Capturando rostro de: {nombre_completo}")
    print(f"[INFO] Se necesitan {FOTOS_OBJETIVO} capturas.")
    print("[INFO] Mira de frente a la cámara.")
    print("[INFO] Presiona ESC para cancelar.\n")

    detector = crear_detector()

    # Crear carpeta para este usuario
    nombre_carpeta = nombre_completo.replace(" ", "_")
    carpeta = os.path.join(DATA_DIR, f"{id_usuario}_{nombre_carpeta}")
    os.makedirs(carpeta, exist_ok=True)

    # Inicializar cámara (soporte para picamera2 en Raspberry Pi)
    if PICAMERA2_DISPONIBLE:
        picam  = Picamera2()
        config = picam.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"})
        picam.configure(config)
        picam.start()
        cap = None
        print("[CAMARA] Usando picamera2 (CSI)")
    else:
        picam = None
        cap   = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] No se pudo abrir la cámara.")
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[CAMARA] Usando OpenCV (USB/webcam)")

    capturadas = 0

    while capturadas < FOTOS_OBJETIVO:
        # Leer frame
        if picam:
            frame = picam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                continue

        frame = cv2.flip(frame, 1)

        # Convertir a escala de grises para la detección
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (0, 0), fx=ESCALA_DETEC, fy=ESCALA_DETEC)

        rostro_coords = detectar_rostro(detector, small)

        if rostro_coords is not None:
            esc = int(1 / ESCALA_DETEC)
            x, y, w, h = rostro_coords
            x, y, w, h = x*esc, y*esc, w*esc, h*esc

            # Recortar y normalizar el rostro
            rostro_crop = gray[y:y+h, x:x+w]
            rostro_res  = cv2.resize(rostro_crop, FACE_SIZE)

            # Guardar imagen
            ruta_img = os.path.join(carpeta, f"rostro_{capturadas:03d}.jpg")
            cv2.imwrite(ruta_img, rostro_res)
            capturadas += 1

            # Dibujar rectángulo en la vista previa
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 212, 170), 2)

        # Barra de progreso
        bw = int((capturadas / FOTOS_OBJETIVO) * 300)
        hf = frame.shape[0]
        cv2.rectangle(frame, (10, hf-30), (310, hf-10), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, hf-30), (10+bw, hf-10), (0, 212, 170), -1)
        cv2.putText(frame, f"{capturadas}/{FOTOS_OBJETIVO} capturas",
                    (10, hf-35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Admin: {nombre_completo}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 170), 2)

        if capturadas >= FOTOS_OBJETIVO:
            cv2.putText(frame, "Listo!",
                        (frame.shape[1]//2 - 50, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 212, 170), 3)

        cv2.imshow("Setup Admin — ESC para cancelar", frame)
        key = cv2.waitKey(1)

        if key == 27:
            print("\n[CANCELADO] Setup cancelado por el usuario.")
            if picam: picam.stop()
            if cap:   cap.release()
            cv2.destroyAllWindows()
            return None

    if picam: picam.stop()
    if cap:   cap.release()
    cv2.destroyAllWindows()

    print(f"[OK] {capturadas} imágenes guardadas en: {carpeta}")
    return carpeta


# ══════════════════════════════════════════════════════════════════════════════
#  GUARDAR EN BD
# ══════════════════════════════════════════════════════════════════════════════

def crear_admin(datos: dict, carpeta_rostros: str) -> int:
    """
    Inserta el admin en la BD y guarda la ruta de sus imágenes.
    Luego entrena el modelo LBPH.
    """
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

    # Guardar ruta de carpeta en datos_biometricos (campo "encoding" reutilizado)
    guardar_encoding(id_usuario, carpeta_rostros)

    # Entrenar modelo LBPH con el primer administrador
    print("[INFO] Entrenando modelo LBPH inicial...")
    entrenar()

    return id_usuario


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*50)
    print("  FACEACCESS — SETUP INICIAL")
    print("="*50)

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

    # Confirmar datos
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

    # Insertar en BD primero para obtener id_usuario
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO usuarios "
            "(nombre, apellido_p, apellido_m, matricula, contrasenia, id_rol) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datos["nombre"], datos["apellido_p"], datos["apellido_m"],
             datos["matricula"], datos["contrasenia"], ROL_ADMIN)
        )
        id_u = cursor.lastrowid

    # Capturar rostro
    carpeta = capturar_rostro(id_u, nombre_completo)
    if carpeta is None:
        print("[ERROR] No se pudo capturar el rostro. Intenta de nuevo.")
        # Revertir inserción
        with get_connection() as conn:
            conn.execute("DELETE FROM usuarios WHERE id_usuario = ?", (id_u,))
        sys.exit(1)

    # Guardar ruta y entrenar
    guardar_encoding(id_u, carpeta)
    print("[INFO] Entrenando modelo LBPH...")
    entrenar()

    print("\n" + "="*50)
    print("  ✓ ADMINISTRADOR CREADO EXITOSAMENTE")
    print("="*50)
    print(f"  Nombre    : {nombre_completo}")
    print(f"  Matrícula : {datos['matricula']}")
    print(f"  ID        : {id_u}")
    print(f"  Imágenes  : {carpeta}")
    print("="*50)
    print("\n[LISTO] Ahora puedes correr ReconocimientoFacial.py")


if __name__ == "__main__":
    main()