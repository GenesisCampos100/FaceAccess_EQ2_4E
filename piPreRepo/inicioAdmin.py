import customtkinter as ctk
from datetime import datetime
import numpy as np
import cv2
import navbarAdmin
import gestor_db
import os
from PIL import Image


from capturar_imagen import procesar_rostro

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

        biometria_registrada = [False]
        modal = ctk.CTkToplevel(padre)
        modal.title("Añadir Nuevo Usuario")
        modal.geometry("450x780")
        modal.attributes("-topmost", True)
        modal.grab_set()
        imagen_capturada = [None]

        ctk.CTkLabel(modal, text="Registro Biométrico", font=("Arial", 22, "bold")).pack(pady=(20, 5))

    
        def cerrar_modal():
            modal.destroy()


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

        ctk.CTkButton(
            modal,
            text="📸 Capturar Rostro",
            fg_color="#3498DB",
            command=lambda: abrir_captura_biometrica(entry_mat.get())
        ).pack(pady=10)

        preview_img = ctk.CTkLabel(modal, text="Sin imagen", width=120, height=120, fg_color="gray")
        preview_img.pack(pady=10)
        
        lbl_err = ctk.CTkLabel(modal, text="", text_color="red"); lbl_err.pack()

        def guardar():
            mat = entry_mat.get()

            if len(mat) != 8 or not entry_nom.get():
                lbl_err.configure(text="⚠️ Datos incompletos")
                return

            if not biometria_registrada[0]:
                lbl_err.configure(text="⚠️ Debe capturar el rostro primero")
                return

            if gestor_db.agregar_usuario(
                mat,
                entry_nom.get(),
                entry_ap.get(),
                entry_am.get(),
                entry_pass.get(),
                roles_map[combo_rol.get()]
            ):
                cerrar_modal()

                if comando_recargar:
                    comando_recargar()
            else:
                lbl_err.configure(text="⚠️ Matrícula duplicada")

        btns = ctk.CTkFrame(modal, fg_color="transparent")
        btns.pack(pady=20)
        ctk.CTkButton(btns, text="Cancelar", fg_color="#E74C3C", command=cerrar_modal).pack(side="left", padx=10)
        ctk.CTkButton(btns, text="Guardar", fg_color="#2ECC71", command=guardar).pack(side="left", padx=10)


        def abrir_captura_biometrica(matricula):

            if len(matricula) != 8:
                lbl_err.configure(text="⚠️ Ingresa matrícula válida primero")
                return

            ventana_cam = ctk.CTkToplevel(modal)
            ventana_cam.title("Captura Biométrica")
            ventana_cam.geometry("400x400")

            label_video = ctk.CTkLabel(ventana_cam, text="")
            label_video.pack(pady=10)

            cap = cv2.VideoCapture(0)
            frame_actual = [None]

            
            def cerrar():
                cap.release()
                ventana_cam.destroy()

                ventana_cam.protocol("WM_DELETE_WINDOW", cerrar)

            def actualizar():
                ret, frame = cap.read()
                if ret:
                    frame_actual[0] = frame.copy()

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    imgtk = ctk.CTkImage(light_image=img, size=(320,240))

                    label_video.configure(image=imgtk)
                    label_video.image = imgtk

                ventana_cam.after(20, actualizar)

            actualizar()

            frame_botones = ctk.CTkFrame(ventana_cam)
            frame_botones.pack(pady=10)

            def capturar():
                if frame_actual[0] is None:
                    return

                procesar_captura(frame_actual[0], matricula)

            def cerrar():
                cap.release()
                ventana_cam.destroy()

            ctk.CTkButton(frame_botones, text="📸 Capturar", command=capturar).pack(side="left", padx=10)
            ctk.CTkButton(frame_botones, text="❌ Cancelar", command=cerrar).pack(side="left", padx=10)

            def procesar_captura(frame, matricula):

                exito, mensaje = procesar_rostro(frame, matricula)

                if exito:
                    lbl_err.configure(text=f"✅ {mensaje}")
                    biometria_registrada[0] = True

                    # 🔥 GUARDAR EN MEMORIA
                    imagen_capturada[0] = frame

                    # 🔥 MOSTRAR EN EL MODAL
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    img_ctk = ctk.CTkImage(light_image=img, size=(120,120))

                    preview_img.configure(image=img_ctk, text="")
                    preview_img.image = img_ctk

                    # 🔥 GUARDAR EN DISCO
                    os.makedirs("capturas", exist_ok=True)
                    cv2.imwrite(f"capturas/{matricula}.jpg", frame)

                    # 🔥 SOLO cerrar ventana de cámara (NO el modal)
                    cap.release()
                    ventana_cam.destroy()

                else:
                    lbl_err.configure(text=f"⚠️ {mensaje}")

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