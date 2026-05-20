import customtkinter as ctk #//! Importar la librería de la interfaz gráfica
from datetime import datetime #//! Importar la librería para manejar fechas y horas
import os

try:
    from PIL import Image, ImageDraw
    PIL_DISPONIBLE = True
except ImportError:
    PIL_DISPONIBLE = False

try:
    import cv2
    CV2_DISPONIBLE = True
except ImportError:
    CV2_DISPONIBLE = False

class MiApp(ctk.CTk): #= Definir la clase MiApp que hereda de ctk.CTk, lo que significa que es una ventana personalizada basada en la clase principal de CustomTkinter
    def __init__(self): #= Método constructor de la clase MiApp, que hereda de ctk.CTk
        super().__init__() #= Llamar al constructor de la clase padre (ctk.CTk) para inicializar la ventana

        # ================= CONFIGURACION DE PAGINA =================
        #- Crear la ventana principal de la aplicación
        self.title("Bienvenido") #= Establecer el título de la ventana
        self.geometry("600x1024") # Establecer el tamaño de la ventana

        #@ Crear los frames para organizar la interfaz
        #- Crear el primer frame (superior)
        self.frame_superior = ctk.CTkFrame(self, fg_color="#7CB9FF", corner_radius=9, height=60, width=600) #= Definimos el frame superior con un color de fondo azul claro, bordes redondeados, altura de 60 y ancho de 600
        self.frame_superior.pack(side="top", fill= "both", padx=5, pady=5) #= Definimos que el frame se empaquete en la parte superior, se expanda para llenar el espacio disponible, y tenga un margen de 5 píxeles alrededor
        self.frame_superior.pack_propagate(False) #= Evitar que el frame inferior ajuste su tamaño automáticamente al contenido
        

        #- Crear el frame intermedio (mediano)
        self.frame_mediano = ctk.CTkFrame(self, fg_color="#6A49FA", corner_radius=9, height=764, width=600) #= Definimos el frame mediano con un color de fondo blanco, bordes redondeados, altura de 864 y ancho de 600
        self.frame_mediano.pack(side="top", fill= "both", padx=5, pady=5) #= Definimos que el frame se empaquete en la parte superior, se expanda para llenar el espacio disponible, y tenga un margen de 5 píxeles alrededor

        #- Crear el frame inferior (inferior)
        self.frame_inferior = ctk.CTkFrame(self, fg_color="#F0BEB4", corner_radius=9, height=200, width=600) #= Definimos el frame inferior con un color de fondo marrón, bordes redondeados, altura de 100 y ancho de 600
        self.frame_inferior.pack(side="top", fill= "both", padx=5, pady=5) #= Definimos que el frame se empaquete en la parte superior, se expanda para llenar el espacio disponible, y tenga un margen de 5 píxeles alrededor
        self.frame_inferior.pack_propagate(False) #= Evitar que el frame inferior ajuste su tamaño automáticamente al contenido

        #@ Añadir los label de cada frame
        self.bloque_hora = ctk.CTkFrame(self.frame_superior, fg_color="transparent") #= Crear un frame para el bloque de hora con un color de fondo transparente, lo que permite que el color del frame superior se muestre a través de él
        self.bloque_hora.pack(side="left", padx=20) #= Empaquetar el bloque de hora en el lado izquierdo del frame superior con un margen de 20 píxeles a la izquierda y a la derecha
        ctk.CTkLabel(self.bloque_hora, text="⏰", font=("Comfortaa", 26), text_color="black").pack(side="left", padx=(0, 8)) #= Agregar un label con un ícono de reloj para indicar la hora, con una fuente Comfortaa de tamaño 26 y color de texto
        self.label_clock = ctk.CTkLabel(self.bloque_hora, text="", font=("Comfortaa", 20, "bold"), text_color="black") #= Crear un label para mostrar la hora actual, con una fuente Comfortaa de tamaño 20 y estilo negrita
        self.label_clock.pack(side="left") #= Empaquetar el label del reloj en el bloque de hora, alineado a la izquierda

        self.bloque_fecha = ctk.CTkFrame(self.frame_superior, fg_color="transparent") #= Crear un frame para el bloque de fecha con un color de fondo transparente, lo que permite que el color del frame superior se muestre a través de él
        self.bloque_fecha.pack(side="right", padx=20) #= Empaquetar el bloque de fecha en el lado derecho del frame superior con un margen de 20 píxeles a la izquierda y a la derecha
        ctk.CTkLabel(self.bloque_fecha, text="📅", font=("Comfortaa", 26), text_color="black").pack(side="left", padx=(0, 8)) #= Agregar un label con un ícono de calendario para indicar la fecha, con una fuente Comfortaa de tamaño 26 y color de texto
        self.label_calendar = ctk.CTkLabel(self.bloque_fecha, text="", font=("Comfortaa", 20, "bold"), text_color="black") #= Crear un label para mostrar la fecha actual, con una fuente Comfortaa de tamaño 20 y estilo negrita
        self.label_calendar.pack(side="left") #= Empaquetar el label del calendario en el bloque de fecha, alineado a la izquierda
        self.actualizar_reloj()

        # ================= CONFIGURACION DE IMAGENES =================
        self.ruta_imagenes = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images")) #= Construir la ruta absoluta a la carpeta "images" que se encuentra en el directorio padre del archivo actual, utilizando os.path.abspath y os.path.join para garantizar la compatibilidad entre sistemas operativos
        self.ruta_foto_default = os.path.join(self.ruta_imagenes, "usuario_demo.jpg") #= Construir la ruta completa a la imagen de usuario predeterminada "usuario_demo.jpg" dentro de la carpeta "images" utilizando os.path.join para garantizar la compatibilidad entre sistemas operativos
        self.imagen_usuario_ctk = None #= Inicializar la variable para almacenar la imagen del usuario en formato compatible con CustomTkinter, inicialmente establecida en None hasta que se cargue una imagen válida
        self._crear_panel_usuario_inferior() #= Llamar al método para crear el panel de usuario en el frame inferior, que incluye la foto del usuario y sus datos (nombre y matrícula)
        self.actualizar_panel_usuario("Sin identificar", "----") #= Actualizar el panel de usuario con un estado inicial que indica que no se ha identificado a ningún usuario, mostrando "Sin identificar" como nombre y "----" como matrícula, y utilizando la imagen predeterminada para el usuario. Este método también se encargará de mostrar la imagen circular del usuario en el label correspondiente.

        # ================= CONFIGURACION DE CAMARA (OpenCV) =================
        self.cap = None #= Inicializar la variable para la captura de video de OpenCV, establecida en None hasta que se inicie la cámara
        self.camera_index = 0 #= Inicializar la variable para el índice de la cámara, establecida en 0, lo que generalmente corresponde a la cámara predeterminada del sistema. Este índice se puede ajustar si se tienen múltiples cámaras conectadas para seleccionar la cámara deseada.
        self.camara_activa = False
        self.imagen_camara_ctk = None
        self.estado_actual = "escaneando"
        self.nombre_estado_actual = ""

        self.label_video = ctk.CTkLabel(self.frame_mediano, text="Iniciando camara...", font=("Comfortaa", 20, "bold"), text_color="white")
        self.label_video.place(relx=0, rely=0, relwidth=1, relheight=1)

        #- Inicialización de vistas de estado
        self._crear_vistas_estado() #= Llamar al método para crear las vistas de estado (escaneando, acceso permitido, acceso denegado) que se mostrarán en el frame mediano según el estado del proceso de reconocimiento facial
        self._mostrar_estado("escaneando") #= Mostrar la vista de estado inicial (escaneando)

        self.after(400, self._iniciar_camara)
        self.protocol("WM_DELETE_WINDOW", self._al_cerrar)

    def _crear_panel_usuario_inferior(self): #= Método para crear el panel de usuario en el frame inferior, que incluye la foto del usuario y sus datos (nombre y matrícula)
        self.label_foto_usuario = ctk.CTkLabel(self.frame_inferior, text="", width=120, height=120) #= Crear un label para mostrar la foto del usuario, con un tamaño de 120x120 píxeles. Inicialmente se establece el texto en vacío, ya que la imagen se cargará posteriormente. Este label se colocará en el lado izquierdo del frame inferior.
        self.label_foto_usuario.pack(side="left", padx=20, pady=10) # = Empaquetar el label de la foto del usuario en el lado izquierdo del frame inferior, con un margen de 20 píxeles a la izquierda y a la derecha, y un margen de 10 píxeles en la parte superior e inferior

        self.frame_datos_usuario = ctk.CTkFrame(self.frame_inferior, fg_color="#D9D9D9", corner_radius=24, width=380, height=120) #= Crear un frame para contener los datos del usuario (nombre y matrícula) con un color de fondo gris claro, bordes redondeados, y un tamaño de 380x120 píxeles. Este frame se colocará a la derecha del label de la foto del usuario dentro del frame inferior.
        self.frame_datos_usuario.pack(side="left", padx=10, pady=10, fill="both", expand=True) #= Empaquetar el frame de datos del usuario en el lado izquierdo del frame inferior, con un margen de 10 píxeles a la izquierda y a la derecha, y un margen de 10 píxeles en la parte superior e inferior. Además, se establece fill="both" para que el frame se expanda tanto horizontal como verticalmente dentro del espacio disponible, y expand=True para permitir que el frame ocupe todo el espacio adicional disponible en el frame inferior.
        self.frame_datos_usuario.pack_propagate(False) #= Evitar que el frame de datos del usuario ajuste su tamaño automáticamente al contenido, lo que permite mantener el tamaño definido de 380x120 píxeles independientemente del contenido que se agregue dentro de él

        self.label_nombre_usuario = ctk.CTkLabel(self.frame_datos_usuario, text="Nombre: ", font=("Comfortaa", 20, "bold"), text_color="black") #= Crear un label para mostrar el nombre del usuario, con un texto inicial de "Nombre: ", una fuente Comfortaa de tamaño 20 y estilo negrita, y color de
        self.label_nombre_usuario.pack(anchor="w", padx=24, pady=(20, 6)) #= Empaquetar el label del nombre del usuario dentro del frame de datos del usuario, alineado a la izquierda (anchor="w"), con un margen de 24 píxeles a la izquierda y a la derecha, y un margen vertical de 20 píxeles en la parte superior y 6 píxeles en la parte inferior para separar el nombre de la matrícula que se mostrará debajo. Este label se actualizará dinámicamente con el nombre del usuario cuando se muestre la vista de acceso permitido.

        self.label_matricula_usuario = ctk.CTkLabel(self.frame_datos_usuario, text="Matricula: ", font=("Comfortaa", 20, "bold"), text_color="black") #= Crear un label para mostrar la matrícula del usuario, con un texto inicial de "Matricula: ", una fuente Comfortaa de tamaño 20 y estilo negrita, y color de texto negro. Este label se colocará debajo del label del nombre del usuario dentro del frame de datos del usuario.
        self.label_matricula_usuario.pack(anchor="w", padx=24, pady=(2, 6)) #= Empaquetar el label de la matrícula del usuario dentro del frame de datos del usuario, alineado a la izquierda (anchor="w"), con un margen de 24 píxeles a la izquierda y a la derecha, y un margen vertical de 2 píxeles en la parte superior y 6 píxeles en la parte inferior para separarla del label del nombre.

    def _crear_imagen_circular(self, ruta_imagen, size=110): #= Método para crear una imagen circular a partir de una imagen dada
        if not PIL_DISPONIBLE: #= Verificar si la biblioteca PIL está disponible
            return None #= Devolver None si PIL no está disponible

        if ruta_imagen and os.path.exists(ruta_imagen): #= Verificar si la ruta de la imagen es válida y el archivo existe
            imagen = Image.open(ruta_imagen).convert("RGBA") # = Abrir la imagen utilizando PIL y convertirla al modo RGBA para asegurarse de que tenga un canal alfa para la transparencia     
        else:
            imagen = Image.new("RGBA", (size, size), (176, 203, 255, 255)) #= Crear una imagen nueva con un fondo azul claro (color RGBA) para usar como imagen predeterminada si la ruta de la imagen no es válida o el archivo no existe

        imagen = imagen.resize((size, size), Image.Resampling.LANCZOS) #= Redimensionar la imagen al tamaño especificado utilizando el método de remuestreo LANCZOS para mantener la calidad de la imagen al reducir su tamaño
        mascara = Image.new("L", (size, size), 0) #= Crear una nueva imagen en modo "L" (escala de grises) para usar como máscara, con el mismo tamaño que la imagen redimension
        draw = ImageDraw.Draw(mascara) # = Crear un objeto de dibujo para la máscara, lo que permite dibujar formas en ella
        draw.ellipse((0, 0, size, size), fill=255) #= Dibujar una elipse (círculo) en la máscara que cubre toda el área de la imagen, rellenándola con el valor 255 (blanco) para crear una máscara circular que se utilizará para recortar la imagen original y hacerla circular

        imagen_circular = Image.new("RGBA", (size, size), (0, 0, 0, 0)) #= Crear una nueva imagen en modo "RGBA" (con canal alfa) para almacenar la imagen circular, con el mismo tamaño que la imagen redimensionada y un fondo transparente
        imagen_circular.paste(imagen, (0, 0), mascara) #= Pegar la imagen redimensionada en la nueva imagen circular utilizando la máscara circular para recortar la imagen original y mantener solo la parte circular visible, mientras que el resto se vuelve transparente
        return ctk.CTkImage(light_image=imagen_circular, dark_image=imagen_circular, size=(size, size)) #= Convertir la imagen circular resultante a un formato compatible con CustomTkinter (CTkImage) para que pueda ser utilizada en los labels de la interfaz gráfica, especificando tanto la imagen para el modo claro como para el modo oscuro, y estableciendo el tamaño de la imagen en el mismo tamaño que se ha definido para la imagen circular. Este método devuelve la imagen circular en formato CTkImage que se puede asignar a un label para mostrarla en la interfaz gráfica.

    def _texto_con_ellipsis(self, texto, max_caracteres): #= Método para recortar un texto y agregar puntos suspensivos si excede el número máximo de caracteres permitido para mostrar en el panel de usuario, asegurando que el texto se ajuste al espacio disponible sin desbordarse
        texto = str(texto).strip() #= Convertir el texto a cadena y eliminar espacios en blanco al principio y al final para asegurarse de que el recorte se realice correctamente sin contar espacios adicionales
        if len(texto) <= max_caracteres: #= Verificar si la longitud del texto es menor o igual al número máximo de caracteres permitido, en cuyo caso se devuelve el texto completo sin recortar ni agregar puntos suspensivos
            return texto #= Devolver el texto completo si no excede el límite de caracteres
        return texto[:max_caracteres - 3].rstrip() + "..." #= Recortar el texto para que tenga espacio para los puntos suspensivos, eliminando cualquier espacio adicional al final del texto recortado, y luego agregar "..." al final para indicar que el texto ha sido truncado. Este método devuelve el texto recortado con puntos suspensivos si excede el límite de caracteres, o el texto completo si no lo excede.

    def actualizar_panel_usuario(self, nombre, matricula, ruta_foto=None): #= Método para actualizar el panel de usuario en el frame inferior con el nombre, matrícula y foto del usuario. Si no se proporciona una ruta de foto válida, se utilizará la imagen predeterminada.
        nombre_recortado = self._texto_con_ellipsis(nombre, 22) #= Recortar el nombre del usuario utilizando el método _texto_con_ellipsis para asegurarse de que no exceda el espacio disponible en el panel de usuario, permitiendo un máximo de 22 caracteres antes de agregar puntos suspensivos si es necesario
        matricula_recortada = self._texto_con_ellipsis(matricula, 22) #= Recortar la matrícula del usuario utilizando el método _texto_con_ellipsis para asegurarse de que no exceda el espacio disponible en el panel de usuario, permitiendo un máximo de 22 caracteres antes de agregar puntos suspensivos si es necesario

        self.label_nombre_usuario.configure(text=f"Nombre: {nombre_recortado}") #= Actualizar el texto del label del nombre del usuario con el nombre recortado, precedido por "Nombre: " para indicar claramente que se trata del nombre del usuario. Este método se encarga de mostrar el nombre del usuario en el panel de usuario, asegurándose de que se ajuste al espacio disponible sin desbordarse.
        self.label_matricula_usuario.configure(text=f"Matricula: {matricula_recortada}") #= Actualizar el texto del label de la matrícula del usuario con la matrícula recortada, precedida por "Matricula: " para indicar claramente que se trata de la matrícula del usuario. Este método se encarga de mostrar la matrícula del usuario en el panel de usuario, asegurándose de que se ajuste al espacio disponible sin desbordarse.

        ruta_final = ruta_foto if ruta_foto else self.ruta_foto_default #= Determinar la ruta final de la foto del usuario, utilizando la ruta proporcionada si es válida, o la ruta de la foto predeterminada si no se proporciona una ruta válida. Esto garantiza que siempre haya una imagen para mostrar en el panel de usuario, incluso si no se ha identificado a ningún usuario o si la ruta de la foto del usuario no es válida. La variable ruta_final se utiliza posteriormente para cargar y mostrar la imagen circular del usuario en el label correspondiente.
        self.imagen_usuario_ctk = self._crear_imagen_circular(ruta_final) #= Crear la imagen circular del usuario utilizando el método _crear_imagen_circular con la ruta final de la foto del usuario, lo que devuelve una imagen en formato CTkImage que se puede asignar al label de la foto del usuario para mostrarla en el panel de usuario. Esta imagen circular se mostrará en el label_foto_usuario, reemplazando el ícono predeterminado si se ha identificado a un usuario con una foto válida, o mostrando el ícono predeterminado si no se ha identificado a ningún usuario o si la ruta de la foto no es válida.

        if self.imagen_usuario_ctk: #= Verificar si se ha creado una imagen circular válida para el usuario, lo que indica que se ha identificado a un usuario con una foto válida o se ha utilizado la foto predeterminada. Si la imagen circular es válida, se muestra en el label de la foto del usuario, reemplazando el ícono predeterminado.
            self.label_foto_usuario.configure(image=self.imagen_usuario_ctk, text="") #= Configurar el label de la foto del usuario para mostrar la imagen circular del usuario, estableciendo el texto en vacío para ocultar el ícono predeterminado. Esto permite que se muestre la foto del usuario en el panel de usuario cuando se ha identificado a un usuario con una foto válida o se ha utilizado la foto predeterminada, proporcionando una representación visual del usuario identificado. Si no se ha identificado a ningún usuario o si la ruta de la foto no es válida, se mostrará el ícono predeterminado en lugar de una imagen circular.
        else:
            self.label_foto_usuario.configure(image=None, text="👤", font=("Comfortaa", 42), text_color="black") #= Configurar el label de la foto del usuario para mostrar el ícono predeterminado, estableciendo el texto en "👤", con una fuente Comfortaa de tamaño 42 y color de texto negro. Esto permite que se muestre el ícono predeterminado en el panel de usuario cuando no se ha identificado a ningún usuario o si la ruta de la foto no es válida.

    #@ Logica para mostrar las vistas de estado
    def _crear_vistas_estado(self): #= Método para crear las vistas de estado
        """El aviso visual se dibuja directamente sobre el video para no tapar la pantalla completa."""

    def _iniciar_camara(self):
        if not CV2_DISPONIBLE:
            self.label_video.configure(text="OpenCV no esta instalado")
            return

        if not PIL_DISPONIBLE:
            self.label_video.configure(text="Pillow no esta instalado")
            return

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.label_video.configure(text="No se pudo abrir la camara")
            return

        self.camara_activa = True
        self._actualizar_frame_camara()

    def _frame_ajustado_a_contenedor(self, frame, ancho_objetivo, alto_objetivo):
        alto, ancho = frame.shape[:2]
        escala = max(ancho_objetivo / ancho, alto_objetivo / alto)

        nuevo_ancho = max(int(ancho * escala), 1)
        nuevo_alto = max(int(alto * escala), 1)
        frame_escalado = cv2.resize(frame, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)

        x_ini = max((nuevo_ancho - ancho_objetivo) // 2, 0)
        y_ini = max((nuevo_alto - alto_objetivo) // 2, 0)
        x_fin = x_ini + ancho_objetivo
        y_fin = y_ini + alto_objetivo
        return frame_escalado[y_ini:y_fin, x_ini:x_fin]

    def _actualizar_frame_camara(self):
        if not self.camara_activa or self.cap is None:
            return

        lectura_ok, frame = self.cap.read()
        if not lectura_ok:
            self.label_video.configure(text="Sin senal de camara")
            self.after(60, self._actualizar_frame_camara)
            return

        frame = cv2.flip(frame, 1)

        ancho_contenedor = max(self.frame_mediano.winfo_width(), 2)
        alto_contenedor = max(self.frame_mediano.winfo_height(), 2)
        frame = self._frame_ajustado_a_contenedor(frame, ancho_contenedor, alto_contenedor)
        frame = self._dibujar_overlay_estado(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen_pil = Image.fromarray(frame_rgb)
        self.imagen_camara_ctk = ctk.CTkImage(light_image=imagen_pil, dark_image=imagen_pil, size=(ancho_contenedor, alto_contenedor))
        self.label_video.configure(image=self.imagen_camara_ctk, text="")

        self.after(30, self._actualizar_frame_camara)

    def _dibujar_overlay_estado(self, frame):
        if self.estado_actual not in ["escaneando", "exito", "denegado"]:
            return frame

        alto, ancho = frame.shape[:2]
        caja_ancho = min(int(ancho * 0.58), 430)
        caja_alto = 120
        x1 = max((ancho - caja_ancho) // 2, 0)
        y1 = max((alto - caja_alto) // 2, 0)
        x2 = min(x1 + caja_ancho, ancho)
        y2 = min(y1 + caja_alto, alto)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame

        roi_blur = cv2.GaussianBlur(roi, (0, 0), 12)
        frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.25, roi_blur, 0.75, 0)

        if self.estado_actual == "escaneando":
            color = (120, 88, 255)
            texto_1 = "Escaneando rostro"
            texto_2 = "Manten tu rostro centrado"
        elif self.estado_actual == "exito":
            color = (62, 190, 120)
            texto_1 = "Acceso permitido"
            texto_2 = self._texto_con_ellipsis(self.nombre_estado_actual or "Usuario identificado", 28)
        else:
            color = (66, 66, 245)
            texto_1 = "Acceso denegado"
            texto_2 = "Usuario no registrado"

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
        frame = cv2.addWeighted(overlay, 0.20, frame, 0.80, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (235, 235, 235), thickness=2)

        fuente = cv2.FONT_HERSHEY_SIMPLEX
        escala_1 = 0.95
        escala_2 = 0.68

        (w1, _), _ = cv2.getTextSize(texto_1, fuente, escala_1, 2)
        (w2, _), _ = cv2.getTextSize(texto_2, fuente, escala_2, 2)

        tx1 = x1 + max((caja_ancho - w1) // 2, 8)
        ty1 = y1 + 46
        tx2 = x1 + max((caja_ancho - w2) // 2, 8)
        ty2 = y1 + 86

        cv2.putText(frame, texto_1, (tx1, ty1), fuente, escala_1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, texto_2, (tx2, ty2), fuente, escala_2, (240, 240, 240), 2, cv2.LINE_AA)
        return frame

    def _detener_camara(self):
        self.camara_activa = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _al_cerrar(self):
        self._detener_camara()
        self.destroy()

    #@ Método para mostrar la vista de estado correspondiente según el estado proporcionado
    """Gestiona el estado actual para dibujar avisos centrados y difuminados sobre el video."""
    def _mostrar_estado(self, estado, nombre_usuario="", matricula_usuario="", ruta_foto_usuario=None): #= Método para mostrar la vista de estado correspondiente según el estado proporcionado, con un parámetro opcional para el nombre del usuario
        if estado == "escaneando":
            self.estado_actual = "escaneando"
            self.nombre_estado_actual = ""

        elif estado == "exito":
            self.estado_actual = "exito"
            self.nombre_estado_actual = nombre_usuario
            self.actualizar_panel_usuario(
                nombre_usuario if nombre_usuario else "Usuario sin nombre",
                matricula_usuario if matricula_usuario else "Sin matricula",
                ruta_foto_usuario
            )
            self.after(3000, lambda: self._mostrar_estado("escaneando")) #= Volver a mostrar la vista de escaneando después de 3 segundos utilizando after()

        elif estado == "denegado":
            self.estado_actual = "denegado"
            self.nombre_estado_actual = ""
            self.actualizar_panel_usuario("No registrado", "----")
            self.after(3000, lambda: self._mostrar_estado("escaneando")) #= Volver a mostrar la vista de escaneando después de 3 segundos utilizando after()

    def actualizar_reloj(self): #= Método para actualizar el reloj y la fecha en los labels correspondientes
        now = datetime.now() #= Obtener la fecha y hora actual utilizando datetime.now() y almacenarla en la variable 'now'
        self.label_clock.configure(text=now.strftime("%H:%M:%S")) #= Actualizar el texto del label del reloj con la hora actual formateada como "HH:MM:SS" utilizando strftime
        self.label_calendar.configure(text=now.strftime("%d/%m/%Y")) #= Actualizar el texto del label de la fecha con la fecha actual formateada como "DD/MM/YYYY" utilizando strftime
        
        #= Se llama a sí misma cada 1000ms (1 segundo)
        self.after(1000, self.actualizar_reloj)
        


if __name__ == "__main__": #= Verificar si el script se está ejecutando directamente (en lugar de ser importado como módulo)
    app = MiApp() #= Crear una instancia de la clase MiApp, lo que inicializa la ventana y sus componentes

    # --- EJEMPLO DE PRUEBA ---
    # Simulamos que a los 4 segundos detecta a alguien con éxito
    app.after(4000, lambda: app._mostrar_estado("exito", "KEVIN", "A01234567"))

    app.mainloop() #= Iniciar el bucle de eventos para mostrar la ventana y permitir la interacción del usuario        