import customtkinter as ctk
import navbarAdmin

def crear_vista(padre, nombre_admin="Usuario", lista_registros=None, comando_cerrar_sesion=None, comando_ir_inicio=None, comando_ir_registros=None):
    if lista_registros is None:
        lista_registros = []

    frame = ctk.CTkFrame(padre, fg_color="#1a202c", corner_radius=0)
    
    # --- 1. NAVBAR ---
    navbarAdmin.crear_navbar(frame, nombre_admin, comando_cerrar_sesion, comando_ir_inicio, comando_ir_registros)

    # ==========================================
    # 2. CONTENIDO PRINCIPAL
    # ==========================================
    content_frame = ctk.CTkFrame(frame, fg_color="transparent")
    content_frame.pack(fill="both", expand=True, padx=40, pady=20)
    
    ctk.CTkLabel(content_frame, text="Registros de Acceso", font=("Arial", 28, "bold"), text_color="white").pack(anchor="w", pady=(0, 15))
    
    tarjeta_blanca = ctk.CTkFrame(content_frame, fg_color="#FFFFFF", corner_radius=15)
    tarjeta_blanca.pack(fill="both", expand=True)
    
    # --- FILTROS ---
    filtros_frame = ctk.CTkFrame(tarjeta_blanca, fg_color="transparent")
    filtros_frame.pack(fill="x", padx=30, pady=(20, 10))
    
    ctk.CTkLabel(filtros_frame, text="Filtros de Búsqueda", font=("Arial", 16, "bold"), text_color="#2C394B").pack(anchor="w", pady=(0, 10))
    
    controles_frame = ctk.CTkFrame(filtros_frame, fg_color="transparent")
    controles_frame.pack(fill="x")
    
    entry_buscar = ctk.CTkEntry(controles_frame, placeholder_text="Buscar por matrícula...", width=250, height=35)
    entry_buscar.pack(side="left", padx=(0, 10))
    
    combo_estado = ctk.CTkOptionMenu(controles_frame, values=["Todos los estados", "Aceptado", "Denegado"], width=150, height=35, fg_color="#3E4A61")
    combo_estado.pack(side="left", padx=(0, 10))
    
    combo_fecha = ctk.CTkOptionMenu(controles_frame, values=["Cualquier fecha", "Hoy", "Ayer", "Últimos 7 días"], width=150, height=35, fg_color="#3E4A61")
    combo_fecha.pack(side="left", padx=(0, 10))
    
    btn_buscar = ctk.CTkButton(controles_frame, text="Buscar 🔍", width=100, height=35, fg_color="#3498DB", hover_color="#2980B9")
    btn_buscar.pack(side="left", padx=(0, 10))
    
    btn_limpiar = ctk.CTkButton(controles_frame, text="Limpiar ✖", width=100, height=35, fg_color="#7F8C8D", hover_color="#95A5A6")
    btn_limpiar.pack(side="left")

    # --- TABLA DE REGISTROS ---
    tabla_frame = ctk.CTkFrame(tarjeta_blanca, fg_color="transparent")
    tabla_frame.pack(fill="both", expand=True, padx=30, pady=(10, 30))
    
    header_tabla = ctk.CTkFrame(tabla_frame, fg_color="#F2F3F4", height=40, corner_radius=8)
    header_tabla.pack(fill="x")
    header_tabla.pack_propagate(False)
    
    w_mat, w_nom, w_fec, w_est = 120, 250, 200, 150
    ctk.CTkLabel(header_tabla, text="Matrícula", font=("Arial", 13, "bold"), text_color="#2C394B", width=w_mat, anchor="w").pack(side="left", padx=(20, 10))
    ctk.CTkLabel(header_tabla, text="Nombre Completo", font=("Arial", 13, "bold"), text_color="#2C394B", width=w_nom, anchor="w").pack(side="left", padx=10)
    ctk.CTkLabel(header_tabla, text="Fecha y Hora", font=("Arial", 13, "bold"), text_color="#2C394B", width=w_fec, anchor="w").pack(side="left", padx=10)
    ctk.CTkLabel(header_tabla, text="Estado", font=("Arial", 13, "bold"), text_color="#2C394B", width=w_est, anchor="center").pack(side="left", padx=10)

    scroll_tabla = ctk.CTkScrollableFrame(tabla_frame, fg_color="transparent")
    scroll_tabla.pack(fill="both", expand=True, pady=(10, 0))

    def renderizar_tabla(registros_a_mostrar):
        for widget in scroll_tabla.winfo_children():
            widget.destroy()
            
        if not registros_a_mostrar:
            ctk.CTkLabel(scroll_tabla, text="No hay registros para mostrar.", font=("Arial", 14, "italic"), text_color="#7F8C8D").pack(pady=50)
            return

        for r in registros_a_mostrar:
            fila = ctk.CTkFrame(scroll_tabla, fg_color="transparent", height=45)
            fila.pack(fill="x", pady=2)
            fila.pack_propagate(False)
            
            ctk.CTkLabel(fila, text=r.get("matricula", "N/A"), font=("Arial", 13), text_color="#2C394B", width=w_mat, anchor="w").pack(side="left", padx=(20, 10))
            ctk.CTkLabel(fila, text=r.get("nombre", "N/A"), font=("Arial", 13), text_color="#2C394B", width=w_nom, anchor="w").pack(side="left", padx=10)
            ctk.CTkLabel(fila, text=r.get("fecha_hora", "N/A"), font=("Arial", 13), text_color="#7F8C8D", width=w_fec, anchor="w").pack(side="left", padx=10)
            
            # --- PILL DE ESTADO (Verde o Rojo) ---
            estado = r.get("estado", "Desconocido")
            color_fondo = "#D5F5E3" if estado == "Aceptado" else "#FADBD8" 
            color_texto = "#27AE60" if estado == "Aceptado" else "#C0392B" 
            
            estado_container = ctk.CTkFrame(fila, fg_color="transparent", width=w_est)
            estado_container.pack(side="left", padx=10)
            estado_container.pack_propagate(False)
            
            pill_frame = ctk.CTkFrame(estado_container, fg_color=color_fondo, corner_radius=12, width=100, height=26)
            pill_frame.pack(expand=True)
            pill_frame.pack_propagate(False)
            ctk.CTkLabel(pill_frame, text=estado, font=("Arial", 12, "bold"), text_color=color_texto).place(relx=0.5, rely=0.5, anchor="center")
            
            ctk.CTkFrame(scroll_tabla, fg_color="#E5E7E9", height=1).pack(fill="x", padx=10, pady=2)

    def aplicar_filtros():
        texto = entry_buscar.get().lower()
        estado_sel = combo_estado.get()
        
        filtrados = []
        for r in lista_registros:
            coincide_texto = texto in str(r.get("matricula", "")).lower()
            coincide_estado = (estado_sel == "Todos los estados") or (r.get("estado", "") == estado_sel)
            
            if coincide_texto and coincide_estado:
                filtrados.append(r)
        
        renderizar_tabla(filtrados)

    def limpiar_filtros():
        entry_buscar.delete(0, 'end')
        combo_estado.set("Todos los estados")
        combo_fecha.set("Cualquier fecha")
        renderizar_tabla(lista_registros)

    btn_buscar.configure(command=aplicar_filtros)
    btn_limpiar.configure(command=limpiar_filtros)

    renderizar_tabla(lista_registros)

    return frame