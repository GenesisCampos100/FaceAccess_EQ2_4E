import customtkinter as ctk

def crear_vista(padre, comando_reintentar, mensaje_error="Autenticación fallida", comando_cancelar=None):
    # Si no le pasamos un comando específico para cancelar, por defecto cerrará la ventana
    if comando_cancelar is None:
        comando_cancelar = padre.destroy

    frame = ctk.CTkFrame(padre, fg_color="#2C394B") 
    
    # Aumentamos un poquito el 'height' a 400 para que quepan ambos botones cómodamente
    tarjeta = ctk.CTkFrame(frame, fg_color="#FFFFFF", corner_radius=15, width=400, height=400)
    tarjeta.place(relx=0.5, rely=0.5, anchor="center") 
    tarjeta.pack_propagate(False) 
    
    # Icono o símbolo de Error
    ctk.CTkLabel(tarjeta, text="❌", font=("Arial", 60)).pack(pady=(30, 10))
    
    # Título de Error
    ctk.CTkLabel(tarjeta, text="¡Acceso Denegado!", font=("Arial", 24, "bold"), text_color="#F44336").pack(pady=(0, 10))
    
    # Mensaje dinámico de por qué falló
    ctk.CTkLabel(tarjeta, text=mensaje_error, font=("Arial", 16), text_color="#7F8C8D", wraplength=300, justify="center").pack(pady=(10, 20))
    
    # ==========================================
    # BOTONES DE ACCIÓN
    # ==========================================
    # Botón Volver a intentar (Rojo)
    btn_reintentar = ctk.CTkButton(tarjeta, text="Reintentar Login", fg_color="#F44336", hover_color="#D32F2F", 
                                  font=("Arial", 14, "bold"), corner_radius=10, height=45, 
                                  command=comando_reintentar)
    btn_reintentar.pack(pady=(10, 10), padx=40, fill="x")

    # Botón Cancelar / Salir (Gris oscuro)
    btn_cancelar = ctk.CTkButton(tarjeta, text="Cancelar y Salir", fg_color="#3E4A61", hover_color="#2a3344", 
                                  font=("Arial", 14, "bold"), corner_radius=10, height=45, 
                                  command=comando_cancelar)
    btn_cancelar.pack(pady=(0, 20), padx=40, fill="x")
    
    return frame