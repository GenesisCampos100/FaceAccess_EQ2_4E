import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import math
from user_viewsTest import UsersView # IMPORTACIÓN DE LA NUEVA VISTA
# --- PALETA DE COLORES ---
COLOR_BG_DARK = "#334155"      # Azul oscuro
COLOR_BG_WHITE = "#FFFFFF"     # Blanco
COLOR_CARD = "#A4BFDA"         # Tarjeta azul claro
COLOR_INPUT_BG = "#FFFFFF"     
COLOR_BTN_REGISTRAR = "#334155" 
COLOR_BTN_ACTION = "#334155"    

# Colores detección
FACE_OK = (0, 255, 0)      # Verde
FACE_BAD = (0, 0, 255)     # Rojo

ctk.set_appearance_mode("light")

class RegistroApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.geometry("1280x720")
        self.title("Registro de Usuario")
        self.configure(fg_color=COLOR_BG_WHITE)

        self.cap = None
        self.camera_running = True
        self.foto_capturada = False
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # --- FONDO DIVIDIDO ---
        
        # Panel Izquierdo (Azul Oscuro)
        self.panel_izquierdo = ctk.CTkFrame(self, fg_color=COLOR_BG_DARK, corner_radius=0)
        self.panel_izquierdo.place(
            relx=0, 
            rely=0, 
            relwidth=0.5,   
            relheight=1
        )

        # Panel Derecho (Blanco)
        self.panel_derecho = ctk.CTkFrame(self, fg_color=COLOR_BG_WHITE, corner_radius=0)
        self.panel_derecho.place(
            relx=0.5,       
            rely=0, 
            relwidth=0.5,   
            relheight=1
        )
        self.init_ui()
        self.iniciar_camara()
        # Al final de __init__
        self.inputs_validados = {"Name": False, "Matricula": False}

    def mostrar_confirmacion(self):
        # 1. Crear la tarjeta con un tamaño más reducido
        self.confirm_card = ctk.CTkFrame(
            self, 
            fg_color="white", 
            corner_radius=20, 
            bg_color="transparent",
            border_width=1.5, 
            border_color="#CBD5E0" 
        )
        
        # AJUSTE DE TAMAÑO: relwidth de 0.4 -> 0.32 | relheight de 0.3 -> 0.22
        self.confirm_card.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.32, relheight=0.22)

        # --- Contenido del modal ---
        # Reducimos el pady (espaciado superior) para ajustarse al nuevo alto
        header_frame = ctk.CTkFrame(self.confirm_card, fg_color="transparent")
        header_frame.pack(pady=(20, 5), padx=15) 

        # Icono e Título un poco más pequeños para ahorrar espacio
        ctk.CTkLabel(header_frame, text="ⓘ", font=("Inter", 24), text_color="#2D3748").pack(side="left", padx=(0, 8))
        ctk.CTkLabel(header_frame, text="¿CONFIRMAR REGISTRO?", font=("Inter", 15, "bold"), text_color="#2D3748").pack(side="left")

        # Texto de apoyo (opcional, lo bajamos a fuente 11)
        ctk.CTkLabel(self.confirm_card, text="Los datos se guardarán permanentemente.", 
                    font=("Inter", 11), text_color="#718096").pack(pady=(0, 15))

        # --- Botones ---
        btn_container = ctk.CTkFrame(self.confirm_card, fg_color="transparent")
        btn_container.pack(fill="x", padx=30) # Reducimos el margen lateral de los botones

        btn_si = ctk.CTkButton(
            btn_container, text="Confirmar", fg_color="#0EAA47", hover_color="#18BB54",
            text_color="white", font=("Inter", 13, "bold"), height=35, corner_radius=13,
            command=self.ejecutar_finalizacion
        )
        btn_si.pack(side="left", expand=True, padx=5)

        btn_no = ctk.CTkButton(
            btn_container, text="Cancelar", fg_color="#DEDEDE", hover_color="#E2E8F0",
            text_color="#4A5568", font=("Inter", 13, "bold"), height=35, corner_radius=13,
            command=self.cerrar_confirmacion
        )
        btn_no.pack(side="left", expand=True, padx=5)

    def init_ui(self):
        # --- LADO IZQUIERDO: FORMULARIO CENTRADO ---
        # Título arriba
        self.title_label = ctk.CTkLabel(
            self.panel_izquierdo, 
            text="REGISTRO DE\nUSUARIO", 
            font=("Inter", 35, "bold"),
            text_color="white",
            justify="left"
        )
        self.title_label.place(relx=0.15, rely=0.1)

        # Tarjeta del formulario (REDUCIDA Y CENTRADA)
        self.form_card = ctk.CTkFrame(self.panel_izquierdo, fg_color=COLOR_CARD, corner_radius=20)
        self.form_card.place(relx=0.4, rely=0.60, anchor="center", relwidth=0.55, relheight=0.70)

        ctk.CTkLabel(self.form_card, text="Ingresa tus datos para continuar", 
                     text_color="#FFFFFF", font=("Inter", 13)).pack(pady=(25, 10))

        self.create_field("Name", "Ejem: KEVIN RAMON")
        self.create_field("Matricula", "Ejem: 2023932303")
        
        # --- MEJORA DEL CAMPO DE ROLES ---
        ctk.CTkLabel(self.form_card, text="Rol", text_color="#1E293B", font=("Inter", 14, "bold")).pack(anchor="w", padx=35)
        
        # Crea un contenedor (Frame) para que el icono parezca estar "fuera" o integrado de forma minimalista
        self.role_container = ctk.CTkFrame(self.form_card, fg_color="transparent")
        self.role_container.pack(fill="x", padx=35, pady=(5, 20))

        self.combo_rol = ctk.CTkComboBox(
        self.role_container, 
        values=["Estudiante", "Profesor", "Administrador"], 
        fg_color=COLOR_INPUT_BG, 
        state="readonly",
        border_width=0,
        corner_radius=12,
        text_color="black",
        variable=ctk.StringVar(value="Seleccione un rol"), 
        font=("Inter", 13),
        
        # --- AJUSTES CRÍTICOS PARA EL DESPLEGABLE ---
        dropdown_font=("Inter", 13),
        dropdown_fg_color="white",          # Fondo del menú (Cambiado a blanco para contraste)
        dropdown_text_color="black",        # Color del texto de las opciones
        dropdown_hover_color="#E2E8F0",     # Color cuando pasas el ratón (un gris muy claro)
        # --------------------------------------------

        height=42,
        button_color=COLOR_INPUT_BG, 
        button_hover_color="#F1F5F9", 
        justify="left"
    )
        # que ocupe todo el ancho del contenedor
        self.combo_rol.pack(fill="x", side="left", expand=True)

        self.btn_registrar = ctk.CTkButton(
            self.form_card, text="Registrar", 
            command=self.mostrar_confirmacion, # <--- CAMBIA ESTO
            hover_color="#0F172A", height=50, font=("Inter", 16, "bold"),
            state="disabled",           # <--- AGREGADO: Empieza bloqueado
            fg_color="#64748B",         # <--- AGREGADO: Color gris inicial
            corner_radius=15,  # <---  para redondear el botón
            border_width=0     # <---  contorno en el botón

        )
        self.btn_registrar.pack(fill="x", padx=35, pady=(10, 20))

        # --- LADO DERECHO: CÁMARA LIMPIA ---
        
        # --- INDICADOR DE ESTADO (Contenedor para separar colores) ---
        self.status_container = ctk.CTkFrame(self.panel_derecho, fg_color="transparent")
        self.status_container.place(relx=0.5, rely=0.08, anchor="center")

        # Icono (Inicia en amarillo/alerta)
        self.status_icon = ctk.CTkLabel(
            self.status_container, 
            text="⚠️", 
            font=("Inter", 20), 
            text_color="#EAB308"
        )
        self.status_icon.pack(side="left", padx=5)

        # Texto (Siempre negro o gris oscuro)
        self.status_text = ctk.CTkLabel(
            self.status_container, 
            text="POR FAVOR CAPTURA TU ROSTRO", 
            font=("Inter", 16, "bold"), 
            text_color="black"
        )
        self.status_text.pack(side="left")

        # 2. EL CONTENEDOR DE VIDEO (Ajustamos rely de 0.45 a 0.48 para dar espacio al texto)
        self.video_container = ctk.CTkFrame(self.panel_derecho, fg_color="transparent")
        self.video_container.place(relx=0.5, rely=0.48, anchor="center", relwidth=0.85, relheight=0.7)

        # Feed de video que llena el espacio
        self.video_label = ctk.CTkLabel(self.video_container, text="", fg_color="#FFFFFF", corner_radius=20)
        self.video_label.pack(fill="both", expand=True)

        # Botones de Acción (El resto se queda igual...)
        self.btn_box = ctk.CTkFrame(self.panel_derecho, fg_color="transparent")
        self.btn_box.place(relx=0.5, rely=0.9, anchor="center", relwidth=0.7)# <--- BAJA este valor (ej. 0.6) para que los botones estén más juntos
        
        # --- 2. MODIFICACIÓN DE BOTÓN "ACEPTAR" ---
        self.btn_aceptar = ctk.CTkButton(
            self.btn_box, 
            text="Aceptar",
            hover_color="#0F172A", 
            fg_color=COLOR_BTN_ACTION, 
            height=45,           # <--- ALTURA: Sube este valor para botones más altos
            width=190,          # <--- ANCHURA: Añade este parámetro para un ancho fijo
            corner_radius=15, 
            font=("Inter", 16, "bold"), # <--- LETRA: Cambia el 16 para el tamaño y "bold" por "normal" si prefieres
            command=self.capturar_foto
        )
        # expand=True hace que el botón intente llenar el espacio del contenedor
        self.btn_aceptar.pack(side="left", expand=True, padx=10)

        # --- BOTÓN "CANCELAR" ---
        self.btn_cancelar = ctk.CTkButton(
            self.btn_box, 
            text="Cancelar", 
            hover_color="#0F172A",
            fg_color=COLOR_BTN_ACTION, 
            height=45,           # <--- ALTURA: Debe ser igual al de Aceptar para simetría
            width=190,          # <--- ANCHURA: Igual al anterior
            corner_radius=15, 
            font=("Inter", 16, "bold"), # <--- LETRA: Ajusta el tamaño aquí también
            command=self.cancelar_app
        )
        self.btn_cancelar.pack(side="left", expand=True, padx=10)

    def create_field(self, label, placeholder):
        # Contenedor para que el mensaje de error no mueva todo el formulario
        container = ctk.CTkFrame(self.form_card, fg_color="transparent")
        container.pack(fill="x", padx=35, pady=2) # Bajé el pady para que quepa el error

        ctk.CTkLabel(container, text=label, text_color="#1E293B", font=("Inter", 14, "bold")).pack(anchor="w")
        
        entry = ctk.CTkEntry(container, placeholder_text=placeholder, fg_color=COLOR_INPUT_BG, 
                             text_color="black", height=40, border_width=1.5, corner_radius=10,
                             placeholder_text_color="gray")
        entry.pack(fill="x")

        # Mensaje de error (invisible al inicio)
        error_msg = ctk.CTkLabel(container, text="", text_color="#555555", font=("Inter", 11))
        error_msg.pack(anchor="w")

        # ASIGNAR VALIDACIÓN: Cada vez que el usuario suelta una tecla
        entry.bind("<KeyRelease>", lambda e: self.validar_input(label, entry, error_msg))
        
        return entry

    def iniciar_camara(self):
        self.cap = cv2.VideoCapture(0)
        self.actualizar_video()

    def actualizar_video(self):
        if self.camera_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                color_dibujo = FACE_BAD
                for (x, y, w, h) in faces:
                    # Validar si el rostro está centrado
                    centro_x = x + (w // 2)
                    if abs(centro_x - (frame.shape[1] // 2)) < 60 and w > 160:
                        color_dibujo = FACE_OK
                    cv2.ellipse(frame, (x + w//2, y + h//2), (w//2, h//2), 0, 0, 360, color_dibujo, 3)

                # REDIMENSIONAR PARA LLENAR EL CONTENEDOR
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # Ajustamos la imagen al ancho del contenedor
                img = img.resize((750, 500), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.configure(image=imgtk)
                self.video_label.image = imgtk
                
        self.after(10, self.actualizar_video)

    def validar_input(self, nombre_campo, widget_entry, widget_error):
        valor = widget_entry.get()
        es_valido = False
        msj = ""

        if nombre_campo == "Matricula":
            if valor.isdigit() and len(valor) >= 8:
                es_valido = True
            elif valor == "": msj = ""
            else: msj = "El campo matricula es obligatorio (solo números)"

        elif nombre_campo == "Name":
            # Valida que sean letras/espacios y más de 3 caracteres
            if len(valor) > 3 and all(x.isalpha() or x.isspace() for x in valor):
                es_valido = True
            elif valor == "": msj = ""
            else: msj = "El campo nombre es obligatorio (solo letras)"

        # Aplicar colores y mensajes
        if es_valido:
            widget_entry.configure(border_color="#29C000") # Verde
            widget_error.configure(text="")
            self.inputs_validados[nombre_campo] = True
        else:
            if valor != "":
                widget_entry.configure(border_color="#FF0000") # Rojo
                widget_error.configure(text=msj)
            else:
                widget_entry.configure(border_color="gray")
                widget_error.configure(text="")
            self.inputs_validados[nombre_campo] = False

        self.checar_boton_registro()

    def iniciar_loading(self):

        self.loading_frames = []

        gif = Image.open("loading.gif")

        try:
            while True:
                frame = gif.copy().resize((42, 42), Image.Resampling.LANCZOS)
                frame = ImageTk.PhotoImage(frame)
                self.loading_frames.append(frame)
                gif.seek(len(self.loading_frames))
        except EOFError:
            pass

        # ocultar icono y texto
        self.status_icon.pack_forget()
        self.status_text.pack_forget()

        # mostrar gif
        self.loading_label = ctk.CTkLabel(self.status_container, text="")
        self.loading_label.pack(side="left", padx=5)

        self.loading_index = 0
        self.animar_gif()

    def animar_gif(self):

        frame = self.loading_frames[self.loading_index]

        self.loading_label.configure(image=frame)

        self.loading_index += 1

        if self.loading_index == len(self.loading_frames):
            self.loading_index = 0

        self.after(80, self.animar_gif)
    
    def mostrar_exito(self):

        self.loading_label.destroy()

        # volver a mostrar icono
        self.status_icon.pack(side="left", padx=5)

        self.status_icon.configure(
            text="✔",
            text_color="#18BB54"
        )

        self.status_text.pack(side="left")
        self.status_text.configure(
            text="REGISTRO COMPLETADO",
            text_color="#18BB54"
        )

        self.after(1500, self.ir_a_lista_usuarios)

    def cerrar_confirmacion(self):
        if hasattr(self, 'confirm_card'):
            self.confirm_card.destroy()

    def ejecutar_finalizacion(self):
        # Aquí va lo que pasaba antes en finalizar_registro
        self.cerrar_confirmacion()
        self.finalizar_registro() # Llama a tu función original que muestra el mensaje de éxito

    def checar_boton_registro(self):
        # Verifica si ambos campos en el diccionario son True
        if all(self.inputs_validados.values()):
            self.btn_registrar.configure(state="normal", fg_color="#334155") # Color original
        else:
            self.btn_registrar.configure(state="disabled", fg_color="#64748B") # Color gris/bloqueado
    
    def capturar_foto(self):
        self.camera_running = False 
        
        
        
        # ACTUALIZAR EL TEXTO (Se mantiene negro pero cambia el mensaje)
        self.status_text.configure(
            text="ROSTRO CAPTURADO",
            text_color="black"
        )
        
        
        # (Opcional)  ocultar los botones de capturar si ya terminó
        self.btn_box.place_forget() 
        
        #self.mostrar_notificacion("Captura exitosa", "#26AE00")

    def cancelar_app(self):
        self.mostrar_notificacion("Cerrando registro", "#FF0B0B")
        self.after(1200, self.destroy)

    def mostrar_notificacion(self, texto, color):
        notif = ctk.CTkFrame(self, fg_color=color, corner_radius=15)
        notif.place(relx=0.5, rely=0.05, anchor="n")
        ctk.CTkLabel(notif, text=texto, text_color="white", font=("Inter", 14, "bold")).pack(padx=20, pady=10)
        self.after(2000, notif.destroy)

    def ir_a_lista_usuarios(self):

        # cerrar cámara
        if self.cap:
            self.cap.release()

        # eliminar todo lo que hay en la ventana
        for widget in self.winfo_children():
            widget.destroy()

        # abrir nueva vista
        UsersView(self)

    def finalizar_registro(self):

        if not self.camera_running:

            self.iniciar_loading()

            # después de 3 segundos mostrar éxito
            self.after(3000, self.mostrar_exito)

        else:
            self.mostrar_notificacion("Captura rostro primero", "#F59E0B")
if __name__ == "__main__":
    app = RegistroApp()
    app.mainloop()