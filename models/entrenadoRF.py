"""
entrenadoRF.py — Entrenamiento y verificación del modelo LBPH.

ARQUITECTURA (corregida):
  - cargar_dataset() lee las imágenes desde DISCO (data_rostros/<id>_<nombre>/).
  - La BD solo se consulta para saber qué usuarios existen y tienen metadato.
  - El modelo .xml sigue guardándose en disco (es el resultado del entrenamiento).
"""

import cv2
import os
import sys
import numpy as np
from database.db_manager import (
    obtener_todos_encodings,
    obtener_usuario_por_id,
    obtener_personas,
    tiene_biometrico,
    DATA_DIR,
)

# ── CAMBIO 1: ruta absoluta para evitar problemas de directorio de trabajo ───
_script_dir  = os.path.dirname(os.path.abspath(__file__))
MODELO_LBPH  = os.path.join(_script_dir, "modelo_lbph.xml")
FACE_SIZE    = (200, 200)
HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def crear_detector_haar():
    detector = cv2.CascadeClassifier(HAAR_CASCADE)
    if detector.empty():
        raise RuntimeError(f"No se pudo cargar: {HAAR_CASCADE}")
    return detector


def verificar_encodings():
    """
    Muestra el estado biométrico de todos los usuarios activos.
    Verifica cuántas imágenes tiene cada uno en disco.
    """
    print("\n" + "="*55)
    print("  VERIFICACIÓN DE IMÁGENES (archivos en disco)")
    print("="*55)

    personas = obtener_personas()

    if not personas:
        print("[AVISO] No hay usuarios activos en la BD.")
        return []

    sin_imagenes = []
    con_imagenes = []

    for id_usuario, nombre in personas:
        carpeta = _buscar_carpeta_usuario(id_usuario)
        if carpeta:
            imgs = [f for f in os.listdir(carpeta)
                    if f.lower().endswith((".jpg", ".png"))]
            con_imagenes.append((id_usuario, nombre))
            print(f"  ✓ {nombre:<30} (ID {id_usuario}) — {len(imgs)} imágenes en disco")
        else:
            sin_imagenes.append((id_usuario, nombre))
            print(f"  ✗ {nombre:<30} (ID {id_usuario}) — Sin imágenes en disco")

    print("="*55)
    print(f"  Con imágenes : {len(con_imagenes)}")
    print(f"  Sin imágenes : {len(sin_imagenes)}")
    print("="*55)

    if sin_imagenes:
        print("\n[AVISO] Los siguientes usuarios necesitan captura de rostro:")
        for id_u, nombre in sin_imagenes:
            print(f"  → {nombre} (ID {id_u}) — Ejecuta capturar_rostro.py")

    return sin_imagenes


def _buscar_carpeta_usuario(id_usuario: int) -> str | None:
    """
    Busca la carpeta de imágenes de un usuario en DATA_DIR.
    La carpeta empieza con '<id_usuario>_'.
    Retorna la ruta completa o None si no existe.
    """
    if not os.path.isdir(DATA_DIR):
        return None
    prefijo = f"{id_usuario}_"
    for nombre in os.listdir(DATA_DIR):
        if nombre.startswith(prefijo):
            ruta = os.path.join(DATA_DIR, nombre)
            if os.path.isdir(ruta):
                return ruta
    return None


def cargar_dataset() -> tuple[list, list]:
    """
    Carga todas las imágenes de todos los usuarios desde disco.

    Lee cada carpeta data_rostros/<id>_<nombre>/ y carga los JPG/PNG
    como arrays numpy en escala de grises.

    Retorna (imagenes, labels):
      - imagenes: lista de arrays numpy 200x200 escala de grises
      - labels:   lista de id_usuario correspondiente a cada imagen
    """
    imagenes = []
    labels   = []

    if not os.path.isdir(DATA_DIR):
        print(f"[AVISO] Carpeta DATA_DIR no existe: {DATA_DIR}")
        return imagenes, labels

    for carpeta_nombre in sorted(os.listdir(DATA_DIR)):
        ruta_carpeta = os.path.join(DATA_DIR, carpeta_nombre)
        if not os.path.isdir(ruta_carpeta):
            continue

        try:
            id_usuario = int(carpeta_nombre.split("_")[0])
        except ValueError:
            print(f"[AVISO] Carpeta con nombre inesperado, se omite: {carpeta_nombre}")
            continue

        archivos = sorted([
            f for f in os.listdir(ruta_carpeta)
            if f.lower().endswith((".jpg", ".png"))
        ])

        if not archivos:
            print(f"[AVISO] Carpeta vacía para ID {id_usuario}: {ruta_carpeta}")
            continue

        for archivo in archivos:
            ruta_img = os.path.join(ruta_carpeta, archivo)
            img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[AVISO] No se pudo leer: {ruta_img}")
                continue
            img_res = cv2.resize(img, FACE_SIZE)
            imagenes.append(img_res)
            labels.append(id_usuario)

    usuarios_unicos = len(set(labels)) if labels else 0
    print(f"[INFO] Dataset cargado: {len(imagenes)} imágenes de {usuarios_unicos} usuarios.")
    return imagenes, labels


def entrenar() -> bool:
    print(f"[LBPH] Iniciando entrenamiento...")
    imagenes, labels = cargar_dataset()

    if not imagenes:
        print("[ERROR] No hay imágenes para entrenar. Captura rostros primero.")
        return False

    if len(set(labels)) < 1:
        print("[ERROR] Se necesita al menos 1 usuario con imágenes.")
        return False

    usuarios_unicos = len(set(labels))
    print(f"[LBPH] Cargadas {len(imagenes)} imágenes de {usuarios_unicos} usuario(s)")

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8)
        print(f"[LBPH] Reconocedor LBPH creado")
    except Exception as e:
        print(f"[ERROR] Error creando reconocedor LBPH: {e}")
        return False

    try:
        recognizer.train(imagenes, np.array(labels, dtype=np.int32))
        print(f"[LBPH] ✓ Entrenamiento completado")
    except Exception as e:
        print(f"[ERROR] Error durante el entrenamiento: {e}")
        return False

    try:
        recognizer.save(MODELO_LBPH)
        if os.path.exists(MODELO_LBPH):
            tamaño = os.path.getsize(MODELO_LBPH)
            print(f"[LBPH] ✓ Modelo guardado y verificado ({tamaño} bytes)")
        else:
            print(f"[ERROR] Archivo NO se guardó en {MODELO_LBPH}")
            return False
        print(f"[LBPH] ✓ {usuarios_unicos} usuario(s) en el modelo")
        return True
    except Exception as e:
        print(f"[ERROR] Error guardando modelo: {e}")
        return False

def cargar_modelo_lbph():
    print(f"[LBPH] Cargando modelo desde: {MODELO_LBPH}")

    if not os.path.exists(MODELO_LBPH):
        print(f"[ERROR] Modelo LBPH no encontrado: {MODELO_LBPH}")
        print("[INFO] Ejecuta entrenadoRF.py para generarlo.")
        return None

    try:
        tamaño = os.path.getsize(MODELO_LBPH)
        print(f"[LBPH] Archivo encontrado ({tamaño} bytes)")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODELO_LBPH)
        print(f"[LBPH] ✓ Modelo cargado en memoria")
        return recognizer
    except Exception as e:
        print(f"[ERROR] Error cargando modelo LBPH: {e}")
        return None


def cargar_encodings_bd():
    recognizer = cargar_modelo_lbph()
    datos      = obtener_todos_encodings()
    ids_list   = [id_u for id_u, _ in datos]
    return recognizer, ids_list


if __name__ == "__main__":
    solo_verificar = "--check" in sys.argv
    sin_imagenes   = verificar_encodings()

    if solo_verificar:
        sys.exit(0)

    if sin_imagenes:
        print("\n[AVISO] Hay usuarios sin imágenes. Solo se entrenarán los que sí tienen.")

    respuesta = input("\n¿Deseas entrenar/re-entrenar el modelo LBPH ahora? (s/n): ").strip().lower()
    if respuesta == "s":
        ok = entrenar()
        if ok:
            print("\n[LISTO] El modelo está listo. Puedes correr ReconocimientoFacial.py")
        else:
            print("\n[ERROR] No se pudo entrenar.")
    else:
        print("[INFO] Entrenamiento omitido.")