import customtkinter as ctk
from datetime import datetime

def crear_navbar(padre, nombre_admin, comando_cerrar_sesion, comando_ir_inicio=None, comando_ir_registros=None):
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
    
    foto_perfil = ctk.CTkFrame(header_frame, fg_color="white", width=50, height=50, corner_radius=25)
    foto_perfil.pack(side="left", padx=10)
    
    textos_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
    textos_frame.pack(side="left", padx=10)
    ctk.CTkLabel(textos_frame, text="¡Bienvenid@ Admin!", font=("Arial", 20, "bold"), text_color="white").pack(anchor="w")
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

    # --- BOTONES DENTRO DEL MENÚ FLOTANTE ---
    btn_ir_usuarios = ctk.CTkButton(menu_flotante, text="👥 Gestión de Usuarios", fg_color="transparent", 
                                    hover_color="#2C394B", anchor="w", font=("Arial", 14), command=comando_ir_inicio)
    btn_ir_usuarios.pack(fill="x", padx=10, pady=(10, 5))

    btn_ir_registros = ctk.CTkButton(menu_flotante, text="📋 Registros de Acceso", fg_color="transparent", 
                                     hover_color="#2C394B", anchor="w", font=("Arial", 14), command=comando_ir_registros)
    btn_ir_registros.pack(fill="x", padx=10, pady=(5, 10))

    return header_frame