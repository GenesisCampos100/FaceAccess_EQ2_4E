import customtkinter as ctk
from PIL import Image
import os

def crear_vista(padre, comando_finalizar, nombre="Usuario", matricula="000000", ruta_foto=None):
    frame = ctk.CTkFrame(padre, fg_color="#2C394B") 
    
    # ==========================================
    # TARJETA CENTRAL BLANCA
    # ==========================================
    tarjeta = ctk.CTkFrame(frame, fg_color="#FFFFFF", corner_radius=15, width=400, height=500)
    tarjeta.place(relx=0.5, rely=0.5, anchor="center") 
    tarjeta.pack_propagate(False) 
    
    # CAMBIO 1: Título actualizado
    ctk.CTkLabel(tarjeta, text="¡Ingreso Exitoso!", font=("Arial", 26, "bold"), text_color="#4CAF50").pack(pady=(40, 20))
    
    # ==========================================
    # MOSTRAR LA FOTO CAPTURADA
    # ==========================================
    if ruta_foto and os.path.exists(ruta_foto):
        img_pil = Image.open(ruta_foto)
        img_ctk = ctk.CTkImage(light_image=img_pil, size=(180, 180))
        lbl_foto = ctk.CTkLabel(tarjeta, text="", image=img_ctk)
        lbl_foto.pack(pady=(10, 20))
    else:
        ctk.CTkLabel(tarjeta, text="[Sin Foto]", width=180, height=180, fg_color="#E0E0E0", text_color="black").pack(pady=(10, 20))
        
    # ==========================================
    # DATOS DEL USUARIO
    # ==========================================
    ctk.CTkLabel(tarjeta, text=f"{nombre}", font=("Arial", 20, "bold"), text_color="#2C394B").pack(pady=(5, 0))
    ctk.CTkLabel(tarjeta, text=f"Matrícula: {matricula}", font=("Arial", 16), text_color="#7F8C8D").pack(pady=(5, 30))
    
    # ==========================================
    # BOTÓN INGRESAR
    # ==========================================
    # CAMBIO 2: Texto del botón actualizado a "Ingresar"
    btn_finalizar = ctk.CTkButton(tarjeta, text="Ingresar", fg_color="#3E4A61", hover_color="#2a3344", 
                                  font=("Arial", 14, "bold"), corner_radius=10, height=45, 
                                  command=comando_finalizar)
    btn_finalizar.pack(pady=(10, 20), padx=40, fill="x")
    
    return frame