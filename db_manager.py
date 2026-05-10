"""
db_manager.py — Módulo central de conexión a la BD para el sistema FaceAccess.
Todas las operaciones con la base de datos pasan por aquí.

"""

import sqlite3
import os
from datetime import datetime, date

# ─── Ruta de la BD ────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "control_acceso.db")

# ─── Constantes de roles ──────────────────────────────────────────────────────
ROL_ADMIN               = 1
ROL_PERSONAL_AUTORIZADO = 2
ROL_PERSONAL_ESCOLAR    = 3
ROL_ALUMNO              = 4

# ─── Carpeta raíz donde se guardan las imágenes de rostros por usuario ────────
# Cada usuario tiene su subcarpeta: DATA_DIR/<id_usuario>_<nombre>/
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_rostros")


def get_connection():
    """Retorna una conexión a la BD con foreign keys activas y timeout."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


# ══════════════════════════════════════════════════════════════════════════════
#  AUTENTICACIÓN Y ROLES
# ══════════════════════════════════════════════════════════════════════════════

def login(matricula: str, contrasenia: str):
    """
    Valida credenciales (matrícula + contraseña).
    Retorna la fila del usuario si es válido y está activo, o None.
    Convierte matrícula a mayúsculas para búsqueda.
    """
    matricula = matricula.upper() if matricula else ""
    with get_connection() as conn:
        usuario = conn.execute(
            "SELECT u.*, r.nombre_rol FROM usuarios u "
            "JOIN roles r ON r.id_rol = u.id_rol "
            "WHERE u.matricula = ? AND u.contrasenia = ? AND u.estatus = 1",
            (matricula, contrasenia)
        ).fetchone()
    return usuario


def puede_registrar(id_rol: int) -> bool:
    """True solo si el rol tiene permiso para registrar usuarios nuevos."""
    return id_rol in (ROL_ADMIN, ROL_PERSONAL_AUTORIZADO)


def roles_asignables(id_rol_sesion: int) -> list:
    """
    Retorna la lista de id_rol que puede asignar el usuario logueado.
    ADMIN               → todos los roles
    PERSONAL_AUTORIZADO → solo PERSONAL_ESCOLAR y ALUMNO
    Cualquier otro      → ninguno
    """
    if id_rol_sesion == ROL_ADMIN:
        return [ROL_ADMIN, ROL_PERSONAL_AUTORIZADO, ROL_PERSONAL_ESCOLAR, ROL_ALUMNO]
    elif id_rol_sesion == ROL_PERSONAL_AUTORIZADO:
        return [ROL_PERSONAL_ESCOLAR, ROL_ALUMNO]
    return []


def nombre_rol(id_rol: int) -> str:
    """Retorna el nombre legible del rol dado su id."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT nombre_rol FROM roles WHERE id_rol = ?", (id_rol,)
        ).fetchone()
    return row["nombre_rol"] if row else "DESCONOCIDO"


# ══════════════════════════════════════════════════════════════════════════════
#  USUARIOS
# ══════════════════════════════════════════════════════════════════════════════

def obtener_personas():
    """
    Retorna lista de (id_usuario, nombre_completo) para todos los usuarios activos.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id_usuario, nombre || ' ' || apellido_p AS nombre_completo "
            "FROM usuarios WHERE estatus = 1"
        ).fetchall()
    return [(r["id_usuario"], r["nombre_completo"]) for r in rows]


def obtener_usuario_por_id(id_usuario: int):
    """Retorna fila completa del usuario o None."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT u.*, r.nombre_rol FROM usuarios u "
            "JOIN roles r ON r.id_rol = u.id_rol "
            "WHERE u.id_usuario = ?", (id_usuario,)
        ).fetchone()


def obtener_usuario_por_matricula(matricula: str):
    """
    Retorna fila completa del usuario por matrícula o None.
    Convierte matrícula a mayúsculas para búsqueda.
    """
    matricula = matricula.upper() if matricula else ""
    with get_connection() as conn:
        return conn.execute(
            "SELECT u.*, r.nombre_rol FROM usuarios u "
            "JOIN roles r ON r.id_rol = u.id_rol "
            "WHERE u.matricula = ? AND u.estatus = 1", (matricula,)
        ).fetchone()


def obtener_usuario_por_nombre(nombre: str, apellido_p: str, apellido_m: str = ""):
    """
    Busca si una persona ya está registrada por nombre y apellidos.
    Retorna la fila del usuario si existe, o None.
    Convierte todos los nombres a mayúsculas para búsqueda consistente.
    """
    nombre = nombre.upper() if nombre else ""
    apellido_p = apellido_p.upper() if apellido_p else ""
    apellido_m = apellido_m.upper() if apellido_m else ""
    
    with get_connection() as conn:
        return conn.execute(
            "SELECT u.*, r.nombre_rol FROM usuarios u "
            "JOIN roles r ON r.id_rol = u.id_rol "
            "WHERE u.nombre = ? AND u.apellido_p = ? AND u.apellido_m = ? AND u.estatus = 1",
            (nombre, apellido_p, apellido_m)
        ).fetchone()


def obtener_personal_autorizado() -> list:
    """Retorna lista de todos los usuarios activos con rol PERSONAL_AUTORIZADO."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM usuarios WHERE id_rol = ? AND estatus = 1",
            (ROL_PERSONAL_AUTORIZADO,)
        ).fetchall()
    return [dict(r) for r in rows]


def registrar_usuario(nombre: str, apellido_p: str, matricula: str,
                      contrasenia: str, id_rol: int = ROL_ALUMNO,
                      apellido_m: str = "", grado: str = "", grupo: str = "") -> int:
    """
    Inserta un usuario nuevo. Retorna su id_usuario.
    Los parámetros grado y grupo son opcionales y se usan principalmente para alumnos.
    
    Usa transacción: si hay error, hace ROLLBACK automático.
    Convierte todos los campos a mayúsculas (excepto contraseña).
    """
    # Convertir a mayúsculas
    nombre = nombre.upper() if nombre else ""
    apellido_p = apellido_p.upper() if apellido_p else ""
    apellido_m = apellido_m.upper() if apellido_m else ""
    matricula = matricula.upper() if matricula else ""
    grado = grado.upper() if grado else ""
    grupo = grupo.upper() if grupo else ""
    
    conn = None
    try:
        conn = get_connection()
        # Transacción explícita
        cursor = conn.execute(
            "INSERT INTO usuarios (nombre, apellido_p, apellido_m, matricula, "
            "contrasenia, id_rol, grado, grupo) VALUES (?,?,?,?,?,?,?,?)",
            (nombre, apellido_p, apellido_m, matricula, contrasenia, id_rol, grado, grupo)
        )
        user_id = cursor.lastrowid
        conn.commit()  # Confirmar transacción
        print(f"[BD] Usuario registrado: {nombre} {apellido_p} (ID {user_id})")
        return user_id
    except Exception as e:
        if conn:
            conn.rollback()  # Deshacer cambios si hay error
        print(f"[ERROR BD] No se pudo registrar usuario: {e}")
        raise  # Re-lanzar excepción para que la maneje ReconocimientoFacial.py


def desactivar_usuario(id_usuario: int):
    """Desactiva un usuario (baja lógica)."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE usuarios SET estatus = 0 WHERE id_usuario = ?",
            (id_usuario,)
        )


# ══════════════════════════════════════════════════════════════════════════════
#  DATOS BIOMÉTRICOS
#  Con LBPH ya NO se guardan vectores de 128 dimensiones.
#  En su lugar, se guarda la ruta de la carpeta de imágenes del usuario.
#  El campo "encoding" de la tabla se reutiliza para almacenar esa ruta.
# ══════════════════════════════════════════════════════════════════════════════

def guardar_encoding(id_usuario: int, ruta_o_datos: str):
    """
    Guarda o actualiza la ruta de carpeta de imágenes del usuario.
    Parámetro reutilizado: antes era el vector JSON, ahora es la ruta en disco.
    Usa UPSERT para no duplicar registros.
    """
    conn = None
    try:
        conn = get_connection()
        conn.execute(
            "INSERT INTO datos_biometricos (id_usuario, encoding, fecha_actualizacion) "
            "VALUES (?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(id_usuario) DO UPDATE SET "
            "encoding=excluded.encoding, fecha_actualizacion=CURRENT_TIMESTAMP",
            (id_usuario, ruta_o_datos)
        )
        conn.commit()
        print(f"[BD] Ruta biométrica guardada para id_usuario {id_usuario}")
    except sqlite3.OperationalError as e:
        print(f"[ERROR BD] No se pudo guardar: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def obtener_encoding(id_usuario: int) -> str | None:
    """Retorna la ruta almacenada del usuario, o None si no existe."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT encoding FROM datos_biometricos WHERE id_usuario = ?",
            (id_usuario,)
        ).fetchone()
    return row["encoding"] if row else None


def obtener_todos_encodings():
    """
    Retorna lista de (id_usuario, ruta) de todos los usuarios activos
    que tienen datos biométricos registrados.
    Mantiene la firma original para compatibilidad.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT db.id_usuario, db.encoding "
            "FROM datos_biometricos db "
            "JOIN usuarios u ON u.id_usuario = db.id_usuario "
            "WHERE u.estatus = 1"
        ).fetchall()
    return [(r["id_usuario"], r["encoding"]) for r in rows]


def tiene_biometrico(id_usuario: int) -> bool:
    """Retorna True si el usuario ya tiene rostro registrado."""
    return obtener_encoding(id_usuario) is not None


def obtener_carpeta_usuario(id_usuario: int) -> str:
    """
    Retorna la ruta de la carpeta de imágenes del usuario.
    Si no existe en BD, la construye a partir del nombre.
    """
    ruta = obtener_encoding(id_usuario)
    if ruta:
        return ruta
    # Construir ruta por defecto
    u = obtener_usuario_por_id(id_usuario)
    if u:
        nombre = f"{u['nombre']}_{u['apellido_p']}"
        return os.path.join(DATA_DIR, f"{id_usuario}_{nombre}")
    return os.path.join(DATA_DIR, str(id_usuario))


# ══════════════════════════════════════════════════════════════════════════════
#  ACCESOS  (entrada / salida)
# ══════════════════════════════════════════════════════════════════════════════

def registrar_entrada(id_usuario: int, metodo: str = "facial") -> int:
    """
    Registra una entrada.
    Si ya hay un acceso abierto hoy para este usuario, retorna ese id_acceso.
    """
    hoy        = date.today().isoformat()
    hora_ahora = datetime.now().strftime("%H:%M:%S")

    with get_connection() as conn:
        existente = conn.execute(
            "SELECT id_acceso FROM accesos "
            "WHERE id_usuario = ? AND fecha = ? AND hora_salida IS NULL",
            (id_usuario, hoy)
        ).fetchone()

        if existente:
            return existente["id_acceso"]

        cursor = conn.execute(
            "INSERT INTO accesos (id_usuario, fecha, hora_entrada, metodo) "
            "VALUES (?, ?, ?, ?)",
            (id_usuario, hoy, hora_ahora, metodo)
        )
        return cursor.lastrowid


def registrar_salida(id_usuario: int) -> bool:
    """
    Cierra el acceso abierto más reciente del usuario hoy.
    Retorna True si encontró un acceso abierto, False si no.
    """
    hoy        = date.today().isoformat()
    hora_ahora = datetime.now().strftime("%H:%M:%S")

    with get_connection() as conn:
        resultado = conn.execute(
            "UPDATE accesos SET hora_salida = ? "
            "WHERE id_usuario = ? AND fecha = ? AND hora_salida IS NULL",
            (hora_ahora, id_usuario, hoy)
        )
        return resultado.rowcount > 0


def tiene_entrada_abierta(id_usuario: int) -> bool:
    """Retorna True si el usuario ya tiene una entrada sin salida registrada hoy."""
    hoy = date.today().isoformat()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id_acceso FROM accesos "
            "WHERE id_usuario = ? AND fecha = ? AND hora_salida IS NULL",
            (id_usuario, hoy)
        ).fetchone()
    return row is not None


# ══════════════════════════════════════════════════════════════════════════════
#  INTENTOS FALLIDOS
# ══════════════════════════════════════════════════════════════════════════════

def registrar_intento_fallido(motivo: str, matricula: str = "",
                               id_usuario: int = None):
    """Guarda un intento de acceso fallido."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO intentos_fallidos (matricula_ingresada, id_usuario, motivo) "
            "VALUES (?, ?, ?)",
            (matricula, id_usuario, motivo)
        )


# ══════════════════════════════════════════════════════════════════════════════
#  EVIDENCIAS
# ══════════════════════════════════════════════════════════════════════════════

def guardar_evidencia(id_acceso: int, url_foto: str, motivo: str = ""):
    """Asocia una foto de evidencia a un registro de acceso."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO evidencias (id_acceso, url_foto, motivo) VALUES (?, ?, ?)",
            (id_acceso, url_foto, motivo)
        )