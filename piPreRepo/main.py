import sys
import os

# --- INICIO DEL PARCHE PARA TKINTER (PYTHON 3.13) ---
os.environ['TCL_LIBRARY'] = r'C:\Users\karol\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\karol\AppData\Local\Programs\Python\Python313\tcl\tk8.6'
# --- FIN DEL PARCHE ---

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
admin_actual_matricula = None # <--- Variable global para rastrear quién está usando la app

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
        comando_finalizar=lambda: mostrar_inicio(nombre, matricula),
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
    # Guardamos el registro fallido (usamos "0000" como genérico)
    gestor_db.registrar_acceso("0000", "Denegado")
    
    marco_actual = admin_Error.crear_vista(
        padre=app, 
        comando_reintentar=mostrar_login,
        mensaje_error=mensaje
    )
    marco_actual.pack(fill="both", expand=True)

def cerrar_sesion(matricula):
    global admin_actual_matricula
    print(f"DEBUG - Cerrando sesión de: {matricula}")
    if matricula:
        gestor_db.registrar_acceso(matricula, "Salida")
    admin_actual_matricula = None # Limpiamos el usuario activo
    mostrar_login()

def mostrar_inicio(nombre, matricula=None): 
    global marco_actual, admin_actual_matricula
    
    admin_actual_matricula = matricula  
    
    if marco_actual is not None:
        marco_actual.destroy()
        
    lista_real = gestor_db.obtener_usuarios()
    
    # === DESCUBRIR ROL DEL USUARIO ROBUSTO ===
    rol_actual = "Admins"
    for u in lista_real:
        if str(u.get("matricula")) == str(matricula):
            r = str(u.get("rol", "ADMIN")).upper()
            if r == "ADMIN": rol_actual = "Admins"
            elif r == "DIRECTIVO": rol_actual = "Directivos"
            elif r in ["PROFESOR", "PROFESORES"]: rol_actual = "Profesores"
            elif r in ["ALUMNO", "ALUMNADO"]: rol_actual = "Alumnado"
            break
            
    print(f"DEBUG - Sesión iniciada por: {nombre} | Rol: {rol_actual}")
        
    marco_actual = inicioAdmin.crear_vista(
        padre=app, 
        nombre_admin=nombre, 
        matricula_admin=matricula,
        rol_admin=rol_actual,
        lista_usuarios=lista_real, 
        comando_recargar=lambda: mostrar_inicio(nombre, matricula),
        comando_cerrar_sesion=lambda: cerrar_sesion(matricula), 
        comando_ir_inicio=lambda: mostrar_inicio(nombre, matricula),
        comando_ir_registros=lambda: mostrar_registros(nombre, matricula, rol_actual)
    )
    marco_actual.pack(fill="both", expand=True)
    
def mostrar_registros(nombre, matricula=None, rol_actual="Admins"): 
    global marco_actual
    if marco_actual is not None:
        marco_actual.destroy()
        
    # Extraemos los registros reales de la BD
    lista_registros_reales = gestor_db.obtener_registros()
        
    marco_actual = registrosAdmin.crear_vista(
        padre=app, 
        nombre_admin=nombre, 
        matricula_admin=matricula,
        lista_registros=lista_registros_reales, 
        comando_cerrar_sesion=lambda: cerrar_sesion(matricula),
        comando_ir_inicio=lambda: mostrar_inicio(nombre, matricula),
        comando_ir_registros=lambda: mostrar_registros(nombre, matricula, rol_actual)
    )
    marco_actual.pack(fill="both", expand=True)

# === FUNCIÓN PARA CUANDO LE DAN A LA "X" DE LA VENTANA ===
def al_cerrar_ventana():
    global admin_actual_matricula
    if admin_actual_matricula:
        print(f"DEBUG - Registrando salida por cierre de ventana: {admin_actual_matricula}")
        gestor_db.registrar_acceso(admin_actual_matricula, "Salida")
    app.destroy()

app.protocol("WM_DELETE_WINDOW", al_cerrar_ventana)

# Iniciar la aplicación en la pantalla de Login
mostrar_login()
app.mainloop()