"""
entrenadoRF.py — Entrenamiento y verificación del modelo LBPH.
"""

import cv2
import os
import sys
import numpy as np
from db_manager import (
    obtener_todos_encodings,
    obtener_usuario_por_id,
    obtener_personas,
    tiene_biometrico,
    DATA_DIR,
)

# ─── Rutas y parámetros ───────────────────────────────────────────────────────
# Usar ruta ABSOLUTA para el modelo para evitar problemas de directorio de trabajo
_script_dir = os.path.dirname(os.path.abspath(__file__))
MODELO_LBPH      = os.path.join(_script_dir, "modelo_lbph.xml")
FACE_SIZE        = (200, 200)          # tamaño al que se normalizan los rostros
HAAR_CASCADE     = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# ─── Crear detector Haar ──────────────────────────────────────────────────────
def crear_detector_haar():
    """
    Crea el detector de rostros con Haar Cascade.
    Haar Cascade es un método clásico basado en características de Haar
    y clasificadores en cascada de Adaboost. No usa redes neuronales.
    """
    detector = cv2.CascadeClassifier(HAAR_CASCADE)
    if detector.empty():
        raise RuntimeError(f"No se pudo cargar: {HAAR_CASCADE}")
    print(f"[INFO] Haar Cascade cargado: {HAAR_CASCADE}")
    return detector


# ─── Verificar estado de imágenes por usuario ────────────────────────────────
def verificar_encodings():
    """
    Muestra el estado de imágenes de todos los usuarios activos.
    En LBPH el 'encoding' equivale a tener imágenes guardadas en disco.
    """
    print("\n" + "="*50)
    print("  VERIFICACIÓN DE IMÁGENES (LBPH)")
    print("="*50)

    personas = obtener_personas()

    if not personas:
        print("[AVISO] No hay usuarios activos en la BD.")
        return []

    sin_imagenes = []
    con_imagenes = []

    for id_usuario, nombre in personas:
        if tiene_biometrico(id_usuario):
            # Verificar que la carpeta exista y tenga imágenes
            from db_manager import obtener_carpeta_usuario
            carpeta = obtener_carpeta_usuario(id_usuario)
            if os.path.isdir(carpeta):
                imgs = [f for f in os.listdir(carpeta)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                if imgs:
                    con_imagenes.append((id_usuario, nombre))
                    print(f"  ✓ {nombre:<30} (ID {id_usuario}) — {len(imgs)} imágenes")
                else:
                    sin_imagenes.append((id_usuario, nombre))
                    print(f"  ✗ {nombre:<30} (ID {id_usuario}) — Carpeta vacía")
            else:
                sin_imagenes.append((id_usuario, nombre))
                print(f"  ✗ {nombre:<30} (ID {id_usuario}) — Sin carpeta en disco")
        else:
            sin_imagenes.append((id_usuario, nombre))
            print(f"  ✗ {nombre:<30} (ID {id_usuario}) — Sin registro biométrico")

    print("="*50)
    print(f"  Con imágenes   : {len(con_imagenes)}")
    print(f"  Sin imágenes   : {len(sin_imagenes)}")
    print("="*50)

    if sin_imagenes:
        print("\n[AVISO] Los siguientes usuarios necesitan captura de rostro:")
        for id_u, nombre in sin_imagenes:
            print(f"  → {nombre} (ID {id_u}) — Ejecuta capturar_rostro.py")

    return sin_imagenes


# ─── Cargar dataset de imágenes para entrenamiento ───────────────────────────
def cargar_dataset() -> tuple[list, list]:
    """
    Recorre todas las carpetas de usuarios y carga sus imágenes como dataset.
    Retorna (imagenes_grises, labels) donde:
      - imagenes_grises: lista de arrays numpy en escala de grises (200x200)
      - labels: lista de id_usuario (int) correspondiente a cada imagen
    """
    print(f"[LBPH] Cargando dataset...")
    datos = obtener_todos_encodings()   # [(id_usuario, ruta_carpeta)]
    imagenes = []
    labels   = []

    print(f"[LBPH] Encontrados {len(datos)} usuario(s) en BD")
    
    for id_usuario, ruta_carpeta in datos:
        if not ruta_carpeta or not os.path.isdir(ruta_carpeta):
            print(f"[LBPH] ⚠ Carpeta no encontrada para ID {id_usuario}: {ruta_carpeta}")
            continue

        archivos = [f for f in os.listdir(ruta_carpeta)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        if not archivos:
            print(f"[LBPH] ⚠ Sin imágenes en carpeta de ID {id_usuario}: {ruta_carpeta}")
            continue

        print(f"[LBPH] Cargando {len(archivos)} imágenes de ID {id_usuario}")
        
        for archivo in archivos:
            ruta_img = os.path.join(ruta_carpeta, archivo)
            img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[LBPH] ⚠ No se pudo leer: {ruta_img}")
                continue
            # Normalizar tamaño para consistencia en el histograma LBP
            img_res = cv2.resize(img, FACE_SIZE)
            imagenes.append(img_res)
            labels.append(id_usuario)   # label = id_usuario (entero)

    usuarios_unicos = len(set(labels))
    print(f"[LBPH] ✓ Dataset cargado: {len(imagenes)} imágenes de {usuarios_unicos} usuario(s)")
    return imagenes, labels


# ─── Entrenar modelo LBPH ────────────────────────────────────────────────────
def entrenar() -> bool:
    """
    Entrena el reconocedor LBPH con todas las imágenes disponibles.
    Guarda el modelo en MODELO_LBPH para que ReconocimientoFacial.py lo cargue.

    LBPH (Local Binary Patterns Histogram):
      - radius=1      → radio del patrón LBP (vecinos a 1 px de distancia)
      - neighbors=8   → 8 vecinos por punto
      - grid_x=8      → 8 celdas horizontales para el histograma
      - grid_y=8      → 8 celdas verticales para el histograma
      - threshold=∞   → sin umbral interno (lo manejamos nosotros al predecir)
    """
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

    # Crear el reconocedor LBPH
    # cv2.face.LBPHFaceRecognizer_create() requiere opencv-contrib-python
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )
        print(f"[LBPH] Reconocedor LBPH creado")
    except Exception as e:
        print(f"[ERROR] Error creando reconocedor LBPH: {e}")
        return False

    # El entrenamiento asigna a cada histograma LBP su label correspondiente
    try:
        print(f"[LBPH] Entrenando con imágenes...")
        recognizer.train(imagenes, np.array(labels, dtype=np.int32))
        print(f"[LBPH] ✓ Entrenamiento completado")
    except Exception as e:
        print(f"[ERROR] Error durante el entrenamiento: {e}")
        return False

    # Guardar en disco para que el sistema de acceso lo cargue al iniciar
    try:
        recognizer.save(MODELO_LBPH)
        print(f"[LBPH] ✓ Modelo guardado en: {MODELO_LBPH}")
        
        # Verificar que el archivo se guardó correctamente
        if os.path.exists(MODELO_LBPH):
            tamaño = os.path.getsize(MODELO_LBPH)
            print(f"[LBPH] ✓ Archivo verificado ({tamaño} bytes)")
        else:
            print(f"[ERROR] Archivo NO se guardó en {MODELO_LBPH}")
            return False
            
        print(f"[LBPH] ✓ {usuarios_unicos} usuario(s) en el modelo")
        return True
    except Exception as e:
        print(f"[ERROR] Error guardando modelo: {e}")
        return False


# ─── Cargar modelo en memoria (para ReconocimientoFacial.py) ─────────────────
def cargar_modelo_lbph():
    """
    Carga el modelo LBPH desde disco.
    Llamado internamente por ReconocimientoFacial.py al iniciar.
    Retorna el reconocedor, o None si no existe el archivo.
    """
    print(f"[LBPH] Cargando modelo desde: {MODELO_LBPH}")
    
    if not os.path.exists(MODELO_LBPH):
        print(f"[ERROR] Modelo LBPH no encontrado: {MODELO_LBPH}")
        print("[INFO] Ejecuta entrenadoRF.py para generar el modelo.")
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


# ─── Compatibilidad: función que antes cargaba encodings en RAM ───────────────
def cargar_encodings_bd():
    """
    Mantiene la firma original para compatibilidad.
    Con LBPH, los 'encodings' son el modelo ya entrenado, no vectores en RAM.
    Retorna (recognizer, ids_list) donde ids_list viene de la BD.
    """
    recognizer = cargar_modelo_lbph()
    datos      = obtener_todos_encodings()
    ids_list   = [id_u for id_u, _ in datos]
    return recognizer, ids_list


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    solo_verificar = "--check" in sys.argv

    sin_imagenes = verificar_encodings()

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
            print("\n[ERROR] No se pudo entrenar. Revisa los mensajes anteriores.")
    else:
        print("[INFO] Entrenamiento omitido.")