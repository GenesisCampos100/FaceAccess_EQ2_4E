import customtkinter as ctk
from datetime import datetime
import navbarAdmin
import gestor_db
import cv2
import os
from PIL import Image

def crear_vista(padre, nombre_admin="Usuario", matricula_admin=None, rol_admin="Admins", lista_usuarios=None, comando_recargar=None, comando_cerrar_sesion=None, comando_ir_inicio=None, comando_ir_registros=None):
    if lista_usuarios is None:
        lista_usuarios = []

    frame = ctk.CTkFrame(padre, fg_color="#1a202c", corner_radius=0)
    
    navbarAdmin.crear_navbar(frame, nombre_admin, matricula_admin, rol_admin, comando_cerrar_sesion, comando_ir_inicio, comando_ir_registros)

    tabs_frame = ctk.CTkFrame(frame, fg_color="transparent")
    tabs_frame.pack(fill="x", pady=(25, 10), padx=30)
    
    # === 1. RESTRICCIÓN DE PESTAÑAS ===
    if rol_admin == "Directivos":
        pestañas = ["Alumnado", "Profesores"]
    elif rol_admin == "Profesores":
        pestañas = ["Alumnado"]
    elif rol_admin == "Alumnado":
        pestañas = ["Mi Perfil"]
    else: # Admins
        pestañas = ["Alumnado", "Profesores", "Directivos", "Admins"]

    for p in pestañas:
        ctk.CTkButton(tabs_frame, text=p, fg_color="#3E4A61", corner_radius=20, height=40).pack(side="left", padx=10, expand=True, fill="x")

    roles_map = {"Admins": 1, "Directivos": 2, "Profesores": 3, "Alumnado": 4}

    header_seccion = ctk.CTkFrame(frame, fg_color="transparent")
    header_seccion.pack(fill="x", padx=30, pady=(10, 0))
    
    titulo = "Usuarios Registrados" if rol_admin != "Alumnado" else "Mi Información"
    ctk.CTkLabel(header_seccion, text=titulo, font=("Arial", 20, "bold"), text_color="white").pack(side="left", padx=10)

    # --- MODAL AÑADIR ---
    def abrir_modal_agregar():
        modal = ctk.CTkToplevel(padre)
        modal.title("Añadir Nuevo Usuario")
        modal.geometry("450x780")
        modal.attributes("-topmost", True)
        modal.grab_set()

        ctk.CTkLabel(modal, text="Registro Biométrico", font=("Arial", 22, "bold")).pack(pady=(20, 5))

        label_camara = ctk.CTkLabel(modal, text="Cargando lente...", width=320, height=240, fg_color="black", corner_radius=10)
        label_camara.pack(pady=10)

        cap_container = [None]
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
                    gray = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2GRAY)
                    rostros = face_cascade.detectMultiScale(gray, 1.3, 5)
                    nombre_txt = entry_nom.get().strip()
                    tag = nombre_txt if nombre_txt else "Encuadre su rostro"

                    for (x, y, w, h) in rostros:
                        cv2.rectangle(frame_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.rectangle(frame_cv, (x, y-30), (x+w, y), (0, 255, 0), -1)
                        cv2.putText(frame_cv, tag, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

                    frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(frame_rgb)
                    img_ctk = ctk.CTkImage(light_image=img_pil, size=(320, 240))
                    label_camara.configure(image=img_ctk, text="")
                    label_camara.image = img_ctk
            modal.after(20, actualizar_frame)

        def iniciar_recursos():
            temp_cap = cv2.VideoCapture(0)
            if not temp_cap.isOpened(): label_camara.configure(text="Error: Cámara ocupada")
            cap_container[0] = temp_cap
            actualizar_frame()

        modal.after(500, iniciar_recursos)

        def v_m(e, w):
            t = w.get(); l = ''.join(c for c in t if c.isdigit())[:8]
            if t != l: w.delete(0, "end"); w.insert(0, l)
        def v_l(e, w):
            t = w.get(); l = ''.join(c for c in t if c.isalpha() or c.isspace())
            if t != l: w.delete(0, "end"); w.insert(0, l)

        entry_mat = ctk.CTkEntry(modal, placeholder_text="Matrícula (8 dígitos)", width=300); entry_mat.pack(pady=5)
        entry_mat.bind("<KeyRelease>", lambda e: v_m(e, entry_mat))
        
        entry_nom = ctk.CTkEntry(modal, placeholder_text="Nombre(s)", width=300); entry_nom.pack(pady=5)
        entry_nom.bind("<KeyRelease>", lambda e: v_l(e, entry_nom))
        
        entry_ap = ctk.CTkEntry(modal, placeholder_text="Apellido Paterno", width=300); entry_ap.pack(pady=5)
        entry_ap.bind("<KeyRelease>", lambda e: v_l(e, entry_ap))
        
        entry_am = ctk.CTkEntry(modal, placeholder_text="Apellido Materno", width=300); entry_am.pack(pady=5)
        entry_am.bind("<KeyRelease>", lambda e: v_l(e, entry_am))
        
        entry_pass = ctk.CTkEntry(modal, placeholder_text="Contraseña", width=300, show="*"); entry_pass.pack(pady=5)
        
        # === RESTRICCIÓN: ROLES PERMITIDOS AL AGREGAR ===
        roles_permitidos = ["Profesores", "Alumnado"] if rol_admin == "Directivos" else list(roles_map.keys())
        combo_rol = ctk.CTkOptionMenu(modal, values=roles_permitidos, width=300); combo_rol.set("Alumnado"); combo_rol.pack(pady=10)
        
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
                lbl_err.configure(text="⚠️ Error al guardar (Duplicado)")

        btns = ctk.CTkFrame(modal, fg_color="transparent")
        btns.pack(pady=20)
        ctk.CTkButton(btns, text="Cancelar", fg_color="#E74C3C", command=cerrar_modal).pack(side="left", padx=10)
        ctk.CTkButton(btns, text="Guardar", fg_color="#2ECC71", command=guardar).pack(side="left", padx=10)
        modal.protocol("WM_DELETE_WINDOW", cerrar_modal)


    # --- MODAL EDITAR ---
    def abrir_modal_editar(usuario_info):
        modal = ctk.CTkToplevel(padre)
        modal.title("Editar Usuario")
        modal.geometry("400x550")
        modal.attributes("-topmost", True)
        modal.grab_set()

        ctk.CTkLabel(modal, text="Editar Datos", font=("Arial", 22, "bold")).pack(pady=(20, 20))
        matricula_actual = usuario_info.get('matricula', '')
        ctk.CTkLabel(modal, text=f"Matrícula: {matricula_actual}", font=("Arial", 14), text_color="gray").pack(pady=(0, 20))

        nombre_completo = usuario_info.get('nombre', '')
        partes = nombre_completo.split()
        nom_inicial = partes[0] if len(partes) > 0 else ""
        ap_inicial = partes[1] if len(partes) > 1 else ""

        entry_nom = ctk.CTkEntry(modal, placeholder_text="Nombre(s)", width=300); entry_nom.insert(0, nom_inicial); entry_nom.pack(pady=10)
        entry_ap = ctk.CTkEntry(modal, placeholder_text="Apellido Paterno", width=300); entry_ap.insert(0, ap_inicial); entry_ap.pack(pady=10)
        entry_am = ctk.CTkEntry(modal, placeholder_text="Apellido Materno", width=300); entry_am.pack(pady=10)
        entry_pass = ctk.CTkEntry(modal, placeholder_text="Nueva Contraseña (Opcional)", width=300, show="*"); entry_pass.pack(pady=10)
        
        # === RESTRICCIÓN: ROLES PERMITIDOS AL EDITAR ===
        roles_permitidos = ["Profesores", "Alumnado"] if rol_admin == "Directivos" else list(roles_map.keys())
        combo_rol = ctk.CTkOptionMenu(modal, values=roles_permitidos, width=300)
        
        rol_txt = usuario_info.get('rol', 'Alumnado').upper()
        if rol_txt == "ADMIN": combo_rol.set("Admins")
        elif rol_txt == "DIRECTIVO": combo_rol.set("Directivos")
        elif rol_txt == "PROFESOR": combo_rol.set("Profesores")
        elif rol_txt == "ALUMNO": combo_rol.set("Alumnado")
        else: combo_rol.set(usuario_info.get('rol', 'Alumnado').capitalize())
        combo_rol.pack(pady=20)
        
        lbl_err = ctk.CTkLabel(modal, text="", text_color="red"); lbl_err.pack()

        def guardar_edicion():
            if not entry_nom.get(): return
            exito = gestor_db.modificar_usuario(matricula_actual, entry_nom.get(), entry_ap.get(), entry_am.get(), entry_pass.get(), roles_map[combo_rol.get()])
            if exito:
                modal.destroy()
                if comando_recargar: comando_recargar()
            else:
                lbl_err.configure(text="⚠️ Error al guardar")

        btns = ctk.CTkFrame(modal, fg_color="transparent")
        btns.pack(pady=20)
        ctk.CTkButton(btns, text="Cancelar", fg_color="#E74C3C", command=modal.destroy).pack(side="left", padx=10)
        ctk.CTkButton(btns, text="Actualizar", fg_color="#F39C12", command=guardar_edicion).pack(side="left", padx=10)


    # === 2. RESTRICCIÓN DE BOTÓN AÑADIR ===
    # Solo los Directivos y Admins pueden ver y usar este botón
    if rol_admin in ["Admins", "Directivos"]:
        ctk.CTkButton(header_seccion, text="⊕ Añadir Usuario", fg_color="#3E4A61", command=abrir_modal_agregar).pack(side="right", padx=10)

    grid = ctk.CTkScrollableFrame(frame, fg_color="transparent", orientation="horizontal", height=320)
    grid.pack(fill="both", expand=True, padx=30, pady=(0, 30))
    
    # === 3. RESTRICCIÓN DE LA LISTA DE USUARIOS ===
    usuarios_filtrados = []
    for u in lista_usuarios:
        r_str = str(u.get("rol", "ALUMNO")).upper()
        
        if rol_admin == "Admins":
            usuarios_filtrados.append(u)
        elif rol_admin == "Directivos" and r_str in ["ALUMNO", "PROFESOR", "PROFESORES"]:
            usuarios_filtrados.append(u)
        elif rol_admin == "Profesores" and r_str in ["ALUMNO", "ALUMNADO"]:
            usuarios_filtrados.append(u)
        elif rol_admin == "Alumnado" and str(u.get("matricula")) == str(matricula_admin):
            usuarios_filtrados.append(u) # El alumno solo se ve a sí mismo
            
    for usuario in usuarios_filtrados:
        mat = usuario.get('matricula', 'N/A')
        card = ctk.CTkFrame(grid, fg_color="#3E4A61", width=200, height=300, corner_radius=15)
        card.pack(side="left", padx=10, pady=5); card.pack_propagate(False)
        
        ruta = f"capturas/{mat}.jpg"
        if os.path.exists(ruta):
            img = ctk.CTkImage(light_image=Image.open(ruta), size=(100, 100))
            ctk.CTkLabel(card, image=img, text="").pack(pady=(15, 10))
        else:
            ctk.CTkFrame(card, fg_color="gray", width=90, height=90).pack(pady=(15, 10))
        
        ctk.CTkLabel(card, text=usuario.get('nombre', 'N/A'), font=("Arial", 13, "bold"), wraplength=160).pack()
        ctk.CTkLabel(card, text=f"ID: {mat}", font=("Arial", 11), text_color="#BDC3C7").pack()
        ctk.CTkLabel(card, text=usuario.get('rol', 'N/A'), font=("Arial", 10, "italic"), text_color="#F1C40F").pack()
        
        btn_f = ctk.CTkFrame(card, fg_color="transparent")
        btn_f.pack(pady=10)
        
        # === 4. RESTRICCIÓN DE EDICIÓN Y ELIMINACIÓN ===
        # Los profesores pueden VER a los alumnos, pero NO los pueden borrar ni editar.
        if rol_admin in ["Admins", "Directivos"]:
            ctk.CTkButton(btn_f, text="✏️", width=40, fg_color="#F39C12", hover_color="#D68910", 
                          command=lambda u=usuario: abrir_modal_editar(u)).pack(side="left", padx=5)
            ctk.CTkButton(btn_f, text="🗑️", width=40, fg_color="#E74C3C", hover_color="#C0392B", 
                          command=lambda m=mat: [gestor_db.eliminar_usuario(m), 
                                                 os.remove(f"capturas/{m}.jpg") if os.path.exists(f"capturas/{m}.jpg") else None, 
                                                 comando_recargar()]).pack(side="left", padx=5)

    return frame