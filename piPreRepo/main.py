import sys
import os

# Forzamos la ruta al .venv que está una carpeta arriba
ruta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ruta_librerias = os.path.join(ruta_base, ".venv", "Lib", "site-packages")

if ruta_librerias not in sys.path:
    sys.path.insert(0, ruta_librerias) # Lo ponemos al inicio para que tenga prioridad


import customtkinter as ctk
import cv2

# Importamos todas tus pantallas y el gestor

import gestor_db
print(f"DEBUG: Usuarios en BD: {gestor_db.obtener_usuarios()}")

import loginAdmin
import admin_Exito
import admin_Error
import inicioAdmin
import registrosAdmin

app = ctk.CTk()
app.title("Sistema Biométrico - Universidad de Colima")
app.geometry("900x600")

marco_actual = None

def mostrar_login():
    global marco_actual
    if marco_actual is not None:
        marco_actual.destroy()
    marco_actual = loginAdmin.crear_vista(app, mostrar_exito, lambda msg: mostrar_error(msg))
    marco_actual.pack(fill="both", expand=True)

def mostrar_exito(nombre, matricula):
    global marco_actual
    if marco_actual is not None:
        marco_actual.destroy()
        
    # Guardamos el registro en la base de datos
    gestor_db.registrar_acceso(matricula, "Aceptado")
        
    ruta_de_la_foto = f"capturas/{matricula}.jpg"
    
    marco_actual = admin_Exito.crear_vista(
        padre=app, 
        comando_finalizar=lambda: mostrar_inicio(nombre), 
        nombre=nombre,
        matricula=matricula,
        ruta_foto=ruta_de_la_foto
    )
    marco_actual.pack(fill="both", expand=True)

def mostrar_error(mensaje="Acceso denegado"):
    global marco_actual
    if marco_actual is not None:
        marco_actual.destroy()
    
    print(f"DEBUG - Motivo de error: {mensaje}")
    # Guardamos el registro fallido (usamos "0000" como genérico si no sabemos quién fue)
    gestor_db.registrar_acceso("0000", "Denegado")
    
    marco_actual = admin_Error.crear_vista(
        padre=app, 
        comando_reintentar=mostrar_login,
        mensaje_error=mensaje
    )
    marco_actual.pack(fill="both", expand=True)

def mostrar_inicio(nombre):
    global marco_actual
    if marco_actual is not None:
        marco_actual.destroy()
        
    lista_real = gestor_db.obtener_usuarios()
        
    marco_actual = inicioAdmin.crear_vista(
        padre=app, 
        nombre_admin=nombre, 
        lista_usuarios=lista_real, 
        comando_recargar=lambda: mostrar_inicio(nombre),
        comando_cerrar_sesion=mostrar_login,
        comando_ir_inicio=lambda: mostrar_inicio(nombre),
        comando_ir_registros=lambda: mostrar_registros(nombre)
    )
    marco_actual.pack(fill="both", expand=True)

def mostrar_registros(nombre):
    global marco_actual
    if marco_actual is not None:
        marco_actual.destroy()
        
    # Extraemos los registros reales de la BD
    lista_registros_reales = gestor_db.obtener_registros()
        
    marco_actual = registrosAdmin.crear_vista(
        padre=app, 
        nombre_admin=nombre, 
        lista_registros=lista_registros_reales, 
        comando_cerrar_sesion=mostrar_login,
        comando_ir_inicio=lambda: mostrar_inicio(nombre),
        comando_ir_registros=lambda: mostrar_registros(nombre)
    )
    marco_actual.pack(fill="both", expand=True)

# Iniciar la aplicación en la pantalla de Login
mostrar_login()
app.mainloop()