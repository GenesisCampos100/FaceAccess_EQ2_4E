import customtkinter as ctk
from datetime import datetime
import navbarAdmin
import gestor_db
import cv2
import os
from PIL import Image

def crear_vista(padre, nombre_admin="Usuario", lista_usuarios=None, comando_recargar=None, comando_cerrar_sesion=None, comando_ir_inicio=None, comando_ir_registros=None):
    if lista_usuarios is None:
        lista_usuarios = []

    frame = ctk.CTkFrame(padre, fg_color="#1a202c", corner_radius=0)
    navbarAdmin.crear_navbar(frame, nombre_admin, comando_cerrar_sesion, comando_ir_inicio, comando_ir_registros)

    # --- PESTAÑAS ---
    tabs_frame = ctk.CTkFrame(frame, fg_color="transparent")
    tabs_frame.pack(fill="x", pady=(25, 10), padx=30)
    
    pestañas = ["Alumnado", "Profesores", "Directivos", "Admins"]
    for p in pestañas:
        ctk.CTkButton(tabs_frame, text=p, fg_color="#3E4A61", corner_radius=20, height=40).pack(side="left", padx=10, expand=True, fill="x")

    roles_map = {"Admins": 1, "Directivos": 2, "Profesores": 3, "Alumnado": 4}
    roles_inversos = {1: "Admins", 2: "Directivos", 3: "Profesores", 4: "Alumnado"}

    # --- MODAL AÑADIR CON SOLUCIÓN DE CÁMARA ---
    def abrir_modal_agregar():
        modal = ctk.CTkToplevel(padre)
        modal.title("Añadir Nuevo Usuario")
        modal.geometry("450x780")
        modal.attributes("-topmost", True)
        modal.grab_set()

        ctk.CTkLabel(modal, text="Registro Biométrico", font=("Arial", 22, "bold")).pack(pady=(20, 5))

        # Contenedor de la cámara
        label_camara = ctk.CTkLabel(modal, text="Cargando lente...", width=320, height=240, fg_color="black", corner_radius=10)
        label_camara.pack(pady=10)

        # Variables de control
        cap_container = [None] # Usamos lista para poder modificarla dentro de funciones
        ultimo_frame = [None]
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        def cerrar_modal():
            if cap_container[0] and cap_container[0].isOpened():
                cap_container[0].release()
            modal.destroy()

        def actualizar_frame():
            if not modal.winfo_exists():
                if cap_container[0]: cap_container[0].release()
                return

            if cap_container[0] and cap_container[0].isOpened():
                ret, frame_cv = cap_container[0].read()
                if ret:
                    ultimo_frame[0] = frame_cv.copy()
                    
                    # Detección y dibujo
                    gray = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2GRAY)
                    rostros = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    nombre_txt = entry_nom.get().strip()
                    tag = nombre_txt if nombre_txt else "Encuadre su rostro"

                    for (x, y, w, h) in rostros:
                        cv2.rectangle(frame_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        # Etiqueta con fondo para legibilidad
                        cv2.rectangle(frame_cv, (x, y-30), (x+w, y), (0, 255, 0), -1)
                        cv2.putText(frame_cv, tag, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

                    frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(frame_rgb)
                    img_ctk = ctk.CTkImage(light_image=img_pil, size=(320, 240))
                    label_camara.configure(image=img_ctk, text="")
                    label_camara.image = img_ctk

            modal.after(20, actualizar_frame)

        # --- INICIO RETARDADO DE CÁMARA (La clave de la solución) ---
        def iniciar_recursos():
            # Intentamos liberar cualquier instancia previa que haya quedado colgada
            temp_cap = cv2.VideoCapture(0)
            if not temp_cap.isOpened():
                label_camara.configure(text="Error: Cámara ocupada por otro proceso")
            cap_container[0] = temp_cap
            actualizar_frame()

        modal.after(500, iniciar_recursos) # Esperamos medio segundo a que el modal se asiente

        # --- VALIDACIONES ---
        def v_m(e, w):
            t = w.get(); l = ''.join(c for c in t if c.isdigit())[:8]
            if t != l: w.delete(0, "end"); w.insert(0, l)
        def v_l(e, w):
            t = w.get(); l = ''.join(c for c in t if c.isalpha() or c.isspace())
            if t != l: w.delete(0, "end"); w.insert(0, l)

        # --- INPUTS ---
        entry_mat = ctk.CTkEntry(modal, placeholder_text="Matrícula (8 dígitos)", width=300); entry_mat.pack(pady=5)
        entry_mat.bind("<KeyRelease>", lambda e: v_m(e, entry_mat))
        
        entry_nom = ctk.CTkEntry(modal, placeholder_text="Nombre(s)", width=300); entry_nom.pack(pady=5)
        entry_nom.bind("<KeyRelease>", lambda e: v_l(e, entry_nom))
        
        entry_ap = ctk.CTkEntry(modal, placeholder_text="Apellido Paterno", width=300); entry_ap.pack(pady=5)
        entry_ap.bind("<KeyRelease>", lambda e: v_l(e, entry_ap))
        
        entry_am = ctk.CTkEntry(modal, placeholder_text="Apellido Materno", width=300); entry_am.pack(pady=5)
        entry_am.bind("<KeyRelease>", lambda e: v_l(e, entry_am))
        
        entry_pass = ctk.CTkEntry(modal, placeholder_text="Contraseña", width=300, show="*"); entry_pass.pack(pady=5)
        
        combo_rol = ctk.CTkOptionMenu(modal, values=list(roles_map.keys()), width=300); combo_rol.set("Alumnado"); combo_rol.pack(pady=10)
        
        lbl_err = ctk.CTkLabel(modal, text="", text_color="red"); lbl_err.pack()

        def guardar():
            mat = entry_mat.get()
            if len(mat) != 8 or not entry_nom.get() or ultimo_frame[0] is None:
                lbl_err.configure(text="⚠️ Datos incompletos o falta rostro")
                return
            
            if gestor_db.agregar_usuario(mat, entry_nom.get(), entry_ap.get(), entry_am.get(), entry_pass.get(), roles_map[combo_rol.get()]):
                os.makedirs("capturas", exist_ok=True)
                cv2.imwrite(f"capturas/{mat}.jpg", ultimo_frame[0])
                cerrar_modal()
                if comando_recargar: comando_recargar()
            else:
                lbl_err.configure(text="⚠️ Error al guardar (Matrícula duplicada)")

        btns = ctk.CTkFrame(modal, fg_color="transparent")
        btns.pack(pady=20)
        ctk.CTkButton(btns, text="Cancelar", fg_color="#E74C3C", command=cerrar_modal).pack(side="left", padx=10)
        ctk.CTkButton(btns, text="Guardar", fg_color="#2ECC71", command=guardar).pack(side="left", padx=10)

        modal.protocol("WM_DELETE_WINDOW", cerrar_modal)

    # --- BOTÓN AÑADIR ---
    ctk.CTkButton(frame, text="⊕ Añadir Usuario", fg_color="#3E4A61", command=abrir_modal_agregar).pack(anchor="e", padx=40, pady=(0, 20))

    # --- GRID DE USUARIOS ---
    grid = ctk.CTkScrollableFrame(frame, fg_color="transparent", orientation="horizontal", height=320)
    grid.pack(fill="both", expand=True, padx=30, pady=(0, 30))
    
    for usuario in lista_usuarios:
        mat = usuario.get('matricula', 'N/A')
        card = ctk.CTkFrame(grid, fg_color="#3E4A61", width=200, height=300, corner_radius=15)
        card.pack(side="left", padx=10, pady=5); card.pack_propagate(False)
        
        # Foto en la tarjeta
        ruta = f"capturas/{mat}.jpg"
        if os.path.exists(ruta):
            img = ctk.CTkImage(light_image=Image.open(ruta), size=(100, 100))
            ctk.CTkLabel(card, image=img, text="").pack(pady=(15, 10))
        else:
            ctk.CTkFrame(card, fg_color="gray", width=90, height=90).pack(pady=(15, 10))
        
        ctk.CTkLabel(card, text=usuario.get('nombre', 'N/A'), font=("Arial", 13, "bold"), wraplength=160).pack()
        ctk.CTkLabel(card, text=f"ID: {mat}", font=("Arial", 11), text_color="#BDC3C7").pack()
        
        btn_f = ctk.CTkFrame(card, fg_color="transparent")
        btn_f.pack(pady=15)
        ctk.CTkButton(btn_f, text="🗑️", width=40, fg_color="#E74C3C", command=lambda m=mat: [gestor_db.eliminar_usuario(m), os.remove(f"capturas/{m}.jpg") if os.path.exists(f"capturas/{m}.jpg") else None, comando_recargar()]).pack()

    return frame