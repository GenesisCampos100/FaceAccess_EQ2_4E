import customtkinter as ctk
import cv2            
from PIL import Image 
import os 
import face_recognition
import threading  

def crear_vista(padre, ir_a_exito, ir_a_error):
    frame = ctk.CTkFrame(padre, fg_color="transparent")
    
    COLOR_FONDO_IZQ = "#2C394B"  
    COLOR_TARJETA = "#9BAED0"    
    COLOR_BOTON = "#3E4A61"      
    COLOR_FONDO_DER = "#FFFFFF"  

    cap = cv2.VideoCapture(0) 
    ultimo_frame_cv = None 
    
    # === NUEVO: Cargamos el modelo rápido de OpenCV para dibujar el rectángulo ===
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detener_camara():
        if cap.isOpened():
            cap.release()

    def ir_exito_seguro(nombre_usuario, matricula_usuario):
        detener_camara() 
        ir_a_exito(nombre_usuario, matricula_usuario)     

    def ir_error_seguro(mensaje):
        detener_camara()
        ir_a_error(mensaje) 

    def cerrar_aplicacion():
        detener_camara()   
        padre.destroy()    

    # ==========================================
    # LÓGICA DE LOGIN CON IA REAL (ASÍNCRONA)
    # ==========================================
    def iniciar_login():
        nombre = entrada_nombre.get().strip()       
        matricula = entrada_matricula.get().strip() 
        
        if nombre == "" or matricula == "":
            lbl_estado.configure(text="¡Faltan datos!", text_color="#F44336")
            return 

        btn_ingresar.configure(state="disabled")
        btn_cancelar.configure(state="disabled")
        lbl_estado.configure(text="Analizando biometría...", text_color="#7E57C2")
        
        ruta_imagen_guardada = f"capturas/{matricula}.jpg"
        ruta_imagen_temporal = "capturas/temp_login.jpg"

        if not os.path.exists(ruta_imagen_guardada):
            print(f"❌ Error: Matrícula {matricula} no encontrada.")
            ir_error_seguro(f"Matrícula {matricula} no registrada.")
            return

        # Aquí guardamos el frame LIMPIO (sin el rectángulo) para que la IA no se confunda
        if ultimo_frame_cv is not None:
            os.makedirs("capturas", exist_ok=True)
            cv2.imwrite(ruta_imagen_temporal, ultimo_frame_cv)
        else:
            ir_error_seguro("Error al capturar la cámara.")
            return

        def procesar_ia():
            rostros_coinciden = False
            mensaje_error = "El rostro no coincide."
            
            try:
                foto_bd = face_recognition.load_image_file(ruta_imagen_guardada)
                foto_vivo = face_recognition.load_image_file(ruta_imagen_temporal)

                encodings_bd = face_recognition.face_encodings(foto_bd)
                encodings_vivo = face_recognition.face_encodings(foto_vivo)

                if len(encodings_bd) == 0:
                    mensaje_error = "Foto guardada inválida."
                elif len(encodings_vivo) == 0:
                    mensaje_error = "No hay rostro en la cámara."
                else:
                    resultado = face_recognition.compare_faces([encodings_bd[0]], encodings_vivo[0], tolerance=0.65)
                    rostros_coinciden = resultado[0]

            except Exception as e:
                print(f"Error procesando imágenes: {e}")
                mensaje_error = "Error procesando imágenes."
            finally:
                if os.path.exists(ruta_imagen_temporal):
                    os.remove(ruta_imagen_temporal)

            if rostros_coinciden:
                frame.after(0, lambda: ir_exito_seguro(nombre, matricula))
            else:
                frame.after(0, lambda: ir_error_seguro(mensaje_error))

        hilo_ia = threading.Thread(target=procesar_ia)
        hilo_ia.daemon = True
        hilo_ia.start()

    # ==========================================
    # INTERFAZ GRÁFICA 
    # ==========================================
    panel_izquierdo = ctk.CTkFrame(frame, fg_color=COLOR_FONDO_IZQ, corner_radius=0, width=400)
    panel_izquierdo.pack(side="left", fill="y")
    panel_izquierdo.pack_propagate(False)

    ctk.CTkLabel(panel_izquierdo, text="INICIO DE SESIÓN\nADMINISTRADOR", font=("Arial", 22, "bold"), text_color="white", justify="left").pack(pady=(40, 20), padx=30, anchor="w")

    tarjeta = ctk.CTkFrame(panel_izquierdo, fg_color=COLOR_TARJETA, corner_radius=15)
    tarjeta.pack(padx=30, pady=10, fill="both", expand=True)

    ctk.CTkLabel(tarjeta, text="Identifíquese en la cámara", font=("Arial", 12, "bold"), text_color="white").pack(pady=(20, 15))

    ctk.CTkLabel(tarjeta, text="Nombre:", font=("Arial", 14, "bold"), text_color="white").pack(anchor="w", padx=20)
    entrada_nombre = ctk.CTkEntry(tarjeta, placeholder_text="Ingrese su nombre...", fg_color="white", text_color="black", corner_radius=10, height=35)
    entrada_nombre.pack(fill="x", padx=20, pady=(0, 15))

    ctk.CTkLabel(tarjeta, text="Matrícula:", font=("Arial", 14, "bold"), text_color="white").pack(anchor="w", padx=20)
    entrada_matricula = ctk.CTkEntry(tarjeta, placeholder_text="Ingrese su matrícula...", fg_color="white", text_color="black", corner_radius=10, height=35)
    entrada_matricula.pack(fill="x", padx=20, pady=(0, 40)) 

    btn_ingresar = ctk.CTkButton(tarjeta, text="Iniciar Sesión", fg_color=COLOR_BOTON, hover_color="#2a3344", font=("Arial", 14, "bold"), corner_radius=10, height=40, command=iniciar_login)
    btn_ingresar.pack(pady=(0, 10), padx=40, fill="x")

    btn_cancelar = ctk.CTkButton(tarjeta, text="Cancelar", fg_color="#F44336", hover_color="#D32F2F", font=("Arial", 14, "bold"), corner_radius=10, height=40, command=cerrar_aplicacion)
    btn_cancelar.pack(pady=(0, 20), padx=40, fill="x")

    panel_derecho = ctk.CTkFrame(frame, fg_color=COLOR_FONDO_DER, corner_radius=0)
    panel_derecho.pack(side="right", fill="both", expand=True)

    frame_titulo_der = ctk.CTkFrame(panel_derecho, fg_color="transparent")
    frame_titulo_der.pack(pady=(40, 20))

    canvas_spinner = ctk.CTkCanvas(frame_titulo_der, width=40, height=40, bg=COLOR_FONDO_DER, highlightthickness=0)
    canvas_spinner.pack(side="left", padx=(0, 15))
    arco = canvas_spinner.create_arc(5, 5, 35, 35, start=0, extent=280, outline="#7E57C2", width=4, style="arc")
    
    lbl_estado = ctk.CTkLabel(frame_titulo_der, text="Esperando cámara...", font=("Arial", 20, "bold"), text_color="black")
    lbl_estado.pack(side="left")

    def animar_spinner(angulo=0):
        if canvas_spinner.winfo_exists():
            nuevo_angulo = (angulo - 10) % 360
            canvas_spinner.itemconfig(arco, start=nuevo_angulo)
            frame.after(30, lambda: animar_spinner(nuevo_angulo))
    animar_spinner()

    espacio_camara = ctk.CTkFrame(panel_derecho, fg_color="#E0E0E0", width=420, height=320, corner_radius=0)
    espacio_camara.pack(pady=10)
    espacio_camara.pack_propagate(False) 
    
    lbl_video = ctk.CTkLabel(espacio_camara, text="")
    lbl_video.pack(expand=True, fill="both")

    def actualizar_frame():
        nonlocal ultimo_frame_cv 
        
        if not lbl_video.winfo_exists():
            return
            
        if cap.isOpened():
            ret, frame_cv = cap.read()
            if ret:
                # 1. Guardamos una copia INTACTA para que la IA la procese después
                ultimo_frame_cv = frame_cv.copy() 
                
                # 2. Convertimos el frame a blanco y negro (procesa súper rápido)
                gray = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2GRAY)
                
                # 3. Detectamos el rostro en tiempo real
                rostros = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                
                # Obtenemos lo que sea que haya escrito en la caja de texto
                nombre_actual = entrada_nombre.get().strip()
                texto_mostrar = nombre_actual if nombre_actual else "Encuadrando..."

                # 4. Dibujamos el cuadrado y el texto SOBRE el frame que ve el usuario
                for (x, y, w, h) in rostros:
                    # Cuadro verde brillante
                    cv2.rectangle(frame_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    # Texto flotante
                    cv2.putText(frame_cv, texto_mostrar, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 5. Pasamos la imagen a la interfaz gráfica
                frame_cv_resized = cv2.resize(frame_cv, (420, 320))
                frame_rgb = cv2.cvtColor(frame_cv_resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                img_ctk = ctk.CTkImage(light_image=img_pil, size=(420, 320))
                
                lbl_video.configure(image=img_ctk)
                lbl_video.image = img_ctk 
        
        frame.after(15, actualizar_frame)

    frame.after(150, actualizar_frame)

    ctk.CTkLabel(panel_derecho, text="Mire directamente a la cámara", font=("Arial", 12), text_color="gray").pack(pady=(5, 20))

    frame.bind("<Destroy>", lambda e: detener_camara())

    return frame