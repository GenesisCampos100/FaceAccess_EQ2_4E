"""
entrenadoRF.py — Verificador de encodings para face_recognition.

Diferencia vs LBPH:
  · Con LBPH este archivo entrenaba un modelo XML cada vez que agregabas alguien.
  · Con face_recognition NO hay modelo que entrenar.
    Los encodings ya se generan en capturar_rostro.py y se guardan en la BD.
  · Este archivo ahora sirve para VERIFICAR que todos los usuarios tienen
    su encoding correcto y está listo para el reconocimiento.

Úsalo para:
  · Confirmar que todos los usuarios tienen encoding en BD antes de correr
    el reconocimiento.
  · Detectar usuarios sin encoding (a quienes les falta capturar rostro).
  · Regenerar encodings desde las fotos guardadas si algo se corrompió.
"""

import cv2
import os
import json
import numpy as np
import face_recognition
from db_manager import (
    obtener_todos_encodings,
    obtener_usuario_por_id,
    obtener_personas,
    tiene_biometrico,
    guardar_encoding,
)

DATA_PATH = "C:/xampp/htdocs/FaceAccess_EQ2_4E/data"


# ─── Verificar estado de encodings ────────────────────────────────────────────
def verificar_encodings():
    """
    Muestra el estado de encodings de todos los usuarios activos.
    """
    print("\n" + "="*50)
    print("  VERIFICACIÓN DE ENCODINGS")
    print("="*50)

    personas = obtener_personas()   # [(id_usuario, nombre_completo)]

    if not personas:
        print("[AVISO] No hay usuarios activos en la BD.")
        return

    sin_encoding    = []
    con_encoding    = []

    for id_usuario, nombre in personas:
        if tiene_biometrico(id_usuario):
            con_encoding.append((id_usuario, nombre))
            print(f"  ✓ {nombre:<30} (ID {id_usuario}) — Encoding OK")
        else:
            sin_encoding.append((id_usuario, nombre))
            print(f"  ✗ {nombre:<30} (ID {id_usuario}) — SIN ENCODING")

    print("="*50)
    print(f"  Con encoding   : {len(con_encoding)}")
    print(f"  Sin encoding   : {len(sin_encoding)}")
    print("="*50)

    if sin_encoding:
        print("\n[AVISO] Los siguientes usuarios no tienen encoding:")
        for id_u, nombre in sin_encoding:
            print(f"  → {nombre} (ID {id_u}) — Ejecuta capturar_rostro.py")

    return sin_encoding


# ─── Regenerar encodings desde fotos guardadas ───────────────────────────────
def regenerar_encoding(id_usuario: int):
    """
    Si el encoding de un usuario se corrompió o perdió,
    lo regenera leyendo las fotos de su carpeta en disco.
    """
    usuario = obtener_usuario_por_id(id_usuario)
    if not usuario:
        print(f"[ERROR] Usuario ID {id_usuario} no encontrado.")
        return False

    nombre      = f"{usuario['nombre']}_{usuario['apellido_p']}"
    folder_name = f"{id_usuario}_{nombre}"
    person_path = os.path.join(DATA_PATH, folder_name)

    if not os.path.exists(person_path):
        print(f"[ERROR] Carpeta no encontrada: {person_path}")
        print("[INFO] Debes ejecutar capturar_rostro.py para este usuario.")
        return False

    imagenes = [f for f in os.listdir(person_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not imagenes:
        print(f"[ERROR] No hay imágenes en {person_path}")
        return False

    print(f"[INFO] Regenerando encoding desde {len(imagenes)} fotos...")
    encodings = []

    for file_name in imagenes:
        ruta   = os.path.join(person_path, file_name)
        imagen = face_recognition.load_image_file(ruta)
        encs   = face_recognition.face_encodings(imagen)
        if encs:
            encodings.append(encs[0])

    if not encodings:
        print("[ERROR] No se detectaron rostros en las fotos guardadas.")
        return False

    encoding_promedio = np.mean(encodings, axis=0)
    encoding_str      = json.dumps(encoding_promedio.tolist())
    guardar_encoding(id_usuario, encoding_str)

    nombre_completo = f"{usuario['nombre']} {usuario['apellido_p']}"
    print(f"[OK] Encoding regenerado para: {nombre_completo}")
    return True


# ─── Cargar todos los encodings en memoria (para el reconocimiento) ───────────
def cargar_encodings_bd() -> tuple[list, list]:
    """
    Carga todos los encodings de la BD en memoria.
    Retorna (encodings_list, ids_list) para usar en face_recognition.compare_faces.
    Llamado internamente por ReconocimientoFacial.py
    """
    datos = obtener_todos_encodings()   # [(id_usuario, encoding_str)]

    encodings_list = []
    ids_list       = []

    for id_usuario, encoding_str in datos:
        try:
            vector = np.array(json.loads(encoding_str))
            encodings_list.append(vector)
            ids_list.append(id_usuario)
        except Exception as e:
            print(f"[AVISO] Encoding inválido para ID {id_usuario}: {e}")

    print(f"[INFO] {len(encodings_list)} encoding(s) cargados desde BD.")
    return encodings_list, ids_list


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sin_encoding = verificar_encodings()

    if sin_encoding:
        respuesta = input("\n¿Deseas regenerar encodings desde fotos guardadas? (s/n): ").strip().lower()
        if respuesta == "s":
            for id_u, nombre in sin_encoding:
                print(f"\n[INFO] Regenerando para: {nombre}")
                regenerar_encoding(id_u)
    else:
        print("\n[OK] Todos los usuarios tienen encoding. Sistema listo.")
