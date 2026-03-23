import customtkinter as ctk
from datetime import datetime
import os
from PIL import Image, ImageDraw

# 👇 SE AGREGÓ 'rol_admin' COMO PARÁMETRO 👇
def crear_navbar(padre, nombre_admin, matricula_admin, rol_admin="Admins", comando_cerrar_sesion=None, comando_ir_inicio=None, comando_ir_registros=None):
    """ Crea y devuelve el frame del encabezado (navbar) con menú desplegable """
    header_frame = ctk.CTkFrame(padre, fg_color="#2C394B", height=80, corner_radius=0)
    header_frame.pack(fill="x", side="top")
    header_frame.pack_propagate(False)
    
    # --- MENÚ DESPLEGABLE (Oculto por defecto) ---
    menu_flotante = ctk.CTkFrame(padre, fg_color="#3E4A61", corner_radius=8, width=220, height=95)
    menu_flotante.pack_propagate(False)
    
    def alternar_menu():
        if menu_flotante.winfo_ismapped():
            menu_flotante.place_forget() # Ocultar
        else:
            menu_flotante.place(x=20, y=85) # Mostrar justo debajo de la navbar
            menu_flotante.lift() # Asegurar que aparezca por encima

    # --- BOTONES DEL HEADER ---
    btn_menu = ctk.CTkButton(header_frame, text="☰", font=("Arial", 24), width=40, fg_color="transparent", hover_color="#3E4A61", command=alternar_menu)
    btn_menu.pack(side="left", padx=(20, 10))
    
    # =========================================================
    # --- FOTO DE PERFIL CIRCULAR ---
    # =========================================================
    ruta_foto = f"capturas/{matricula_admin}.jpg"
    
    if matricula_admin and os.path.exists(ruta_foto):
        try:
            # 1. Cargar imagen
            img = Image.open(ruta_foto).convert("RGBA")
            
            # 2. Recortar al centro para que sea cuadrada (evita que se deforme)
            min_side = min(img.size)
            left = (img.width - min_side) / 2
            top = (img.height - min_side) / 2
            right = (img.width + min_side) / 2
            bottom = (img.height + min_side) / 2
            img = img.crop((left, top, right, bottom))
            img = img.resize((50, 50))
            
            # 3. Crear una máscara circular
            mask = Image.new('L', (50, 50), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, 50, 50), fill=255)
            
            # 4. Aplicar la máscara a la imagen
            img_circular = Image.new('RGBA', (50, 50), (0, 0, 0, 0))
            img_circular.paste(img, (0, 0), mask)
            
            # 5. Mostrarla en la interfaz
            foto_ctk = ctk.CTkImage(light_image=img_circular, size=(50, 50))
            lbl_foto = ctk.CTkLabel(header_frame, image=foto_ctk, text="")
            lbl_foto.pack(side="left", padx=10)
            
        except Exception as e:
            print(f"Error al procesar la foto de perfil: {e}")
            # Si hay error, mostrar círculo blanco
            foto_perfil = ctk.CTkFrame(header_frame, fg_color="white", width=50, height=50, corner_radius=25)
            foto_perfil.pack(side="left", padx=10)
    else:
        # Si no existe el archivo de foto, mostrar círculo blanco por defecto
        foto_perfil = ctk.CTkFrame(header_frame, fg_color="white", width=50, height=50, corner_radius=25)
        foto_perfil.pack(side="left", padx=10)
    # =========================================================
    
    # --- TEXTOS DE BIENVENIDA ---
    textos_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
    textos_frame.pack(side="left", padx=10)
    
    # 👇 CAMBIO: Saludo personalizado según el rol 👇
    saludo_txt = "¡Bienvenid@!" if rol_admin == "Alumnado" else "¡Bienvenid@ Admin!"
    ctk.CTkLabel(textos_frame, text=saludo_txt, font=("Arial", 20, "bold"), text_color="white").pack(anchor="w")
    ctk.CTkLabel(textos_frame, text=f"*{nombre_admin}*", font=("Arial", 14), text_color="#9BAED0").pack(anchor="w")
    
    btn_logout = ctk.CTkButton(header_frame, text="Cerrar sesión 🚪", fg_color="#E74C3C", hover_color="#C0392B", 
                               width=120, height=35, corner_radius=15, font=("Arial", 13, "bold"),
                               command=comando_cerrar_sesion)
    btn_logout.pack(side="right", padx=(10, 30))

    ahora = datetime.now()
    hora_str = ahora.strftime("%I:%M %p")
    fecha_str = ahora.strftime("%d/%m/%y")
    
    info_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
    info_frame.pack(side="right", padx=10)
    ctk.CTkLabel(info_frame, text=f"🕒  {hora_str}", font=("Arial", 14), text_color="white").pack(anchor="e")
    ctk.CTkLabel(info_frame, text=f"📅  {fecha_str}", font=("Arial", 14), text_color="white").pack(anchor="e")

    # =========================================================
    # --- RESTRICCIÓN DE BOTONES EN EL MENÚ FLOTANTE ---
    # =========================================================
    
    # Si NO es alumno, puede ver "Gestión de Usuarios" (Admins, Directivos, Profesores)
    if rol_admin != "Alumnado":
        btn_ir_usuarios = ctk.CTkButton(menu_flotante, text="👥 Gestión de Usuarios", fg_color="transparent", 
                                        hover_color="#2C394B", anchor="w", font=("Arial", 14), command=comando_ir_inicio)
        btn_ir_usuarios.pack(fill="x", padx=10, pady=(10, 5))

    # Solo los Admins y Directivos pueden ver los Registros de Acceso
    if rol_admin in ["Admins", "Directivos"]:
        btn_ir_registros = ctk.CTkButton(menu_flotante, text="📋 Registros de Acceso", fg_color="transparent", 
                                         hover_color="#2C394B", anchor="w", font=("Arial", 14), command=comando_ir_registros)
        btn_ir_registros.pack(fill="x", padx=10, pady=(5, 10))

    return header_frame