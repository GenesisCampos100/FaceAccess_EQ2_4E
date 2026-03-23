import sqlite3
import os

# ==========================================
# CONFIGURACIÓN DE RUTA (EL GPS MÁGICO)
# ==========================================
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DB_NOMBRE = os.path.join(DIRECTORIO_ACTUAL, "control_acceso.db")

def obtener_usuarios():
    usuarios_lista = []
    try:
        conexion = sqlite3.connect(DB_NOMBRE)
        cursor = conexion.cursor()
        consulta = """
            SELECT u.nombre || ' ' || u.apellido_p AS nombre_completo, u.matricula, r.nombre_rol 
            FROM usuarios u
            LEFT JOIN roles r ON u.id_rol = r.id_rol
        """
        cursor.execute(consulta)
        filas = cursor.fetchall()
        for fila in filas:
            usuarios_lista.append({
                "nombre": fila[0] if fila[0] else "Sin Nombre",
                "matricula": fila[1] if fila[1] else "000000",
                "rol": fila[2] if fila[2] else "Sin Rol"
            })
    except sqlite3.Error as error:
        print(f"Error al leer la base de datos: {error}")
    finally:
        if 'conexion' in locals() and conexion:
            conexion.close()
    return usuarios_lista

def agregar_usuario(matricula, nombre, apellido_p, apellido_m, contrasenia, id_rol):
    try:
        conexion = sqlite3.connect(DB_NOMBRE)
        cursor = conexion.cursor()
        consulta = """
            INSERT INTO usuarios (matricula, nombre, apellido_p, apellido_m, contrasenia, id_rol) 
            VALUES (?, ?, ?, ?, ?, ?)
        """
        cursor.execute(consulta, (matricula, nombre, apellido_p, apellido_m, contrasenia, id_rol))
        conexion.commit()
        return True
    except sqlite3.Error as error:
        print(f"Error al insertar el usuario: {error}")
        return False
    finally:
        if 'conexion' in locals() and conexion:
            conexion.close()

def verificar_admin(matricula, contrasenia):
    try:
        conexion = sqlite3.connect(DB_NOMBRE)
        cursor = conexion.cursor()
        consulta = "SELECT nombre, apellido_p FROM usuarios WHERE matricula = ? AND contrasenia = ? AND id_rol = 4"
        cursor.execute(consulta, (matricula, contrasenia))
        resultado = cursor.fetchone()
        if resultado:
            return True, f"{resultado[0]} {resultado[1]}"
        return False, None
    except sqlite3.Error as error:
        print(f"Error al verificar admin: {error}")
        return False, None
    finally:
        if 'conexion' in locals() and conexion:
            conexion.close()

def eliminar_usuario(matricula):
    try:
        conexion = sqlite3.connect(DB_NOMBRE)
        cursor = conexion.cursor()
        cursor.execute("DELETE FROM usuarios WHERE matricula = ?", (matricula,))
        conexion.commit()
        return True
    except sqlite3.Error as error:
        print(f"Error al eliminar usuario: {error}")
        return False
    finally:
        if 'conexion' in locals() and conexion:
            conexion.close()

def obtener_usuario_por_matricula(matricula):
    conn = sqlite3.connect('control_acceso.db')
    cursor = conn.cursor()
    
    # Limpiamos la matrícula de espacios y aseguramos que sea texto
    matricula_limpia = str(matricula).strip()
    
    # Buscamos usando la matrícula limpia
    cursor.execute("SELECT nombre, apellido_p FROM usuarios WHERE CAST(matricula AS TEXT) = ?", (matricula_limpia,))
    
    resultado = cursor.fetchone()
    conn.close()
    return resultado

def modificar_usuario(matricula, nombre, apellido_p, apellido_m, contrasenia, id_rol):
    try:
        conexion = sqlite3.connect(DB_NOMBRE)
        cursor = conexion.cursor()
        
        # Si la contraseña viene vacía, actualizamos todo MENOS la contraseña
        if contrasenia.strip() == "":
            consulta = """
                UPDATE usuarios 
                SET nombre = ?, apellido_p = ?, apellido_m = ?, id_rol = ?
                WHERE matricula = ?
            """
            cursor.execute(consulta, (nombre, apellido_p, apellido_m, id_rol, matricula))
        else:
            # Si escribieron algo, actualizamos también la contraseña
            consulta = """
                UPDATE usuarios 
                SET nombre = ?, apellido_p = ?, apellido_m = ?, contrasenia = ?, id_rol = ?
                WHERE matricula = ?
            """
            cursor.execute(consulta, (nombre, apellido_p, apellido_m, contrasenia, id_rol, matricula))
            
        conexion.commit()
        return True
    except sqlite3.Error as error:
        print(f"Error al modificar el usuario: {error}")
        return False
    finally:
        if 'conexion' in locals() and conexion:
            conexion.close()

# ==========================================
# GESTIÓN DE REGISTROS DE ACCESO
# ==========================================
def crear_tabla_registros():
    try:
        conexion = sqlite3.connect(DB_NOMBRE)
        cursor = conexion.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS registros_acceso (
                id_registro INTEGER PRIMARY KEY AUTOINCREMENT,
                matricula TEXT NOT NULL,
                fecha_hora DATETIME DEFAULT CURRENT_TIMESTAMP,
                estado TEXT NOT NULL
            )
        """)
        conexion.commit()
    except sqlite3.Error as error:
        print(f"Error al crear tabla de registros: {error}")
    finally:
        if 'conexion' in locals() and conexion:
            conexion.close()

def registrar_acceso(matricula, estado):
    crear_tabla_registros()
    try:
        conexion = sqlite3.connect(DB_NOMBRE)
        cursor = conexion.cursor()
        consulta = "INSERT INTO registros_acceso (matricula, fecha_hora, estado) VALUES (?, datetime('now', 'localtime'), ?)"
        cursor.execute(consulta, (matricula, estado))
        conexion.commit()
    except sqlite3.Error as error:
        print(f"Error al guardar el registro: {error}")
    finally:
        if 'conexion' in locals() and conexion:
            conexion.close()

def obtener_registros():
    crear_tabla_registros()
    registros_lista = []
    try:
        conexion = sqlite3.connect(DB_NOMBRE)
        cursor = conexion.cursor()
        consulta = """
            SELECT r.matricula, u.nombre || ' ' || u.apellido_p AS nombre_completo, r.fecha_hora, r.estado
            FROM registros_acceso r
            LEFT JOIN usuarios u ON r.matricula = u.matricula
            ORDER BY r.fecha_hora DESC
        """
        cursor.execute(consulta)
        filas = cursor.fetchall()
        for fila in filas:
            registros_lista.append({
                "matricula": fila[0],
                "nombre": fila[1] if fila[1] else "Usuario Desconocido",
                "fecha_hora": fila[2],
                "estado": fila[3]
            })
    except sqlite3.Error as error:
        print(f"Error al leer los registros: {error}")
    finally:
        if 'conexion' in locals() and conexion:
            conexion.close()
    return registros_lista