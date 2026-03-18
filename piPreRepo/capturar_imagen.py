import cv2                          #camara
import face_recognition             #reconocimiento facial
import sqlite3                      #base de datos
import numpy as np                  #manejo de vectores
import time                         #contador


from gestor_db import (
    obtener_encodings,
    guardar_biometria,
)


'''
>>  Flujo del sistema

Ingresar usuario
Consultar base
Encender cámara
Detectar rostro
Generar encoding
Comparar encodings
    
¿Existe duplicado?

SI → cancelar registro
NO → guardar encoding1

'''

# Conexión a la base de datos

#Ingresa usuario (Aquí se puede agregar el formulario o modificar el 
# código para que se ejecute después de ingresar el usuario en el registro)

matricula = input("Ingrese matrícula del usuario: ")


#Se obtienen los encodings de la base de datos para compararlos con el nuevo encoding que se va a generar

datos = obtener_encodings()

encodings_guardados = []

for fila in datos:
    encoding = np.frombuffer(fila[0], dtype=np.float64)
    encodings_guardados.append(encoding)


#Activa la camara para capturar la imagen del rostro
cap = cv2.VideoCapture(0)


#Inicializa el contador para registrar el rostro
contador_activo = False
inicio_contador = 0
duracion_contador = 3



# Captura de imagen y detección de rostro en tiempo real
while True:

    ret, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rostros = face_recognition.face_locations(rgb)


    #Dibuja un rectángulo alrededor de cada rostro detectado en la imagen
    for (top, right, bottom, left) in rostros:
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0,255,0),
            2
        )


    #Agrega una instruccion en pantalla para el usuario
    cv2.putText(
        frame,
        "Coloque su rostro frente a la camara",
        (30,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )


    cv2.imshow("Registro biometrico", frame)

    #Debe haber solo un rostro en pantalla, si hay mas de uno la camara sigue prendida hasta que registre solo uno.

    if len(rostros) == 1:

        if not contador_activo:
            inicio_contador = time.time()
            contador_activo = True

        tiempo_restante = duracion_contador - int(time.time() - inicio_contador)

        if tiempo_restante > 0:

            cv2.putText(
                frame,
                str(tiempo_restante),
                (300,200),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,
                (0,0,255),
                4
            )

        else:
            print("Capturando rostro...")
            break

    elif len(rostros) > 1:

        contador_activo = False
        print("Hay más de un rostro en pantalla")

    else:

        contador_activo = False


    cv2.imshow("Registro biometrico", frame)

    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()



#Genera el nuevo encoding a partir de la imagen capturada del usuario de quien se hace registro
encoding_nuevo = face_recognition.face_encodings(rgb, rostros)[0]


#Verifica que no haya duplicados en la base de datos comparando el nuevo encoding con los encodings guardados
duplicado = False


duplicado = False

for encoding in encodings_guardados:

    resultado = face_recognition.compare_faces(
        [encoding],
        encoding_nuevo,
        tolerance=0.5
    )

    if resultado[0]:
        duplicado = True
        break


#Si existe un duplicado, se cancela el registro y se cierra la aplicación. Si no existe, se guarda el nuevo encoding en la base de datos.
if duplicado:

    print("Este rostro ya está registrado en el sistema")

else:

    encoding_bytes = encoding_nuevo.tobytes()

    if guardar_biometria(matricula, encoding_bytes):
        print("Biometría registrada correctamente")
    else:
        print("Error al guardar biometría")


#Se cierran camara, ventanas y conexión a la base de datos
cap.release()
cv2.destroyAllWindows()