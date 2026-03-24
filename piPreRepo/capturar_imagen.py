import cv2                          #camara
import face_recognition             #reconocimiento facial
import sqlite3                      #base de datos
import numpy as np                  #manejo de vectores
import time                         #contador


from gestor_db import (
    obtener_encodings,
    guardar_biometria,
)

def procesar_rostro(frame, matricula):
    import face_recognition
    import numpy as np
    from gestor_db import obtener_encodings, guardar_biometria
    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rostros = face_recognition.face_locations(rgb)

    # Validación: solo 1 rostro
    if len(rostros) != 1:
        return False, "Debe haber un solo rostro"

    encoding_nuevo = face_recognition.face_encodings(rgb, rostros)[0]

    # Obtener encodings existentes
    datos = obtener_encodings()
    encodings_guardados = [
        np.frombuffer(fila[0], dtype=np.float64)
        for fila in datos
    ]

    # Validar duplicados
    for enc in encodings_guardados:
        if face_recognition.compare_faces([enc], encoding_nuevo, tolerance=0.5)[0]:
            return False, "Rostro ya registrado"

    # Guardar
    if guardar_biometria(matricula, encoding_nuevo.tobytes()):
        return True, "Biometría registrada correctamente"
    else:
        return False, "Error al guardar biometría"