"""
db_manager.py — Módulo central de conexión a la BD para el sistema FaceAccess.
Todas las operaciones con la base de datos pasan por aquí.

ARQUITECTURA BIOMÉTRICA (corregida):
  - Las imágenes del rostro se guardan en DISCO: data_rostros/<id>_<nombre>/
  - La BD solo guarda METADATOS: quién tiene biométrico, dónde y desde cuándo.
  - La tabla imagenes_biometricas (BLOB) fue eliminada — es innecesaria con LBPH.
  - datos_biometricos.encoding almacena la RUTA de la carpeta en disco.
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
# Estructura: DATA_DIR/<id_usuario>_<nombre>_<apellido>/rostro_000.jpg ...
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
    Convierte matrícula a mayúsculas para búsqueda (mejora: evita errores por minúsculas).
    """
    # CAMBIO: normalizar a mayúsculas antes de buscar en BD
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
    # CAMBIO: normalizar a mayúsculas antes de buscar en BD
    matricula = matricula.upper() if matricula else ""
    with get_connection() as conn:
        return conn.execute(
            "SELECT u.*, r.nombre_rol FROM usuarios u "
            "JOIN roles r ON r.id_rol = u.id_rol "
            "WHERE u.matricula = ? AND u.estatus = 1", (matricula,)
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
    Los parámetros grado y grupo son opcionales (principalmente para alumnos).

    CAMBIOS respecto a versión anterior:
      - Todos los campos de texto se normalizan a mayúsculas (excepto contraseña).
      - Se agregaron los campos grado y grupo al INSERT.
      - Transacción explícita con rollback si ocurre algún error.
    """
    # CAMBIO: normalizar todos los campos a mayúsculas
    nombre     = nombre.upper()     if nombre     else ""
    apellido_p = apellido_p.upper() if apellido_p else ""
    apellido_m = apellido_m.upper() if apellido_m else ""
    matricula  = matricula.upper()  if matricula  else ""
    grado      = grado.upper()      if grado      else ""
    grupo      = grupo.upper()      if grupo      else ""

    conn = None
    try:
        conn = get_connection()
        # CAMBIO: transacción explícita con commit/rollback
        cursor = conn.execute(
            "INSERT INTO usuarios (nombre, apellido_p, apellido_m, matricula, "
            "contrasenia, id_rol, grado, grupo) VALUES (?,?,?,?,?,?,?,?)",
            (nombre, apellido_p, apellido_m, matricula, contrasenia, id_rol, grado, grupo)
        )
        user_id = cursor.lastrowid
        conn.commit()
        print(f"[BD] Usuario registrado: {nombre} {apellido_p} (ID {user_id})")
        return user_id
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[ERROR BD] No se pudo registrar usuario: {e}")
        raise  # Re-lanzar para que lo maneje quien llame a esta función


def desactivar_usuario(id_usuario: int):
    """Desactiva un usuario (baja lógica)."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE usuarios SET estatus = 0 WHERE id_usuario = ?",
            (id_usuario,)
        )


# ══════════════════════════════════════════════════════════════════════════════
#  DATOS BIOMÉTRICOS — SOLO METADATOS EN BD, IMÁGENES EN DISCO
#
#  Las imágenes del rostro viven en disco: data_rostros/<id>_<nombre>/
#  La BD (tabla datos_biometricos) solo registra:
#    · id_usuario          → a quién pertenece
#    · encoding            → ruta de la carpeta en disco (referencia)
#    · fecha_actualizacion → cuándo se registró o actualizó por última vez
#
#  ¿Por qué disco y no BLOB?
#    LBPH necesita todas las imágenes crudas en RAM para re-entrenarse.
#    Guardarlas como BLOB en SQLite implica deserializarlas todas cada vez,
#    inflando la BD y haciendo el entrenamiento más lento, especialmente
#    en Raspberry Pi. Los archivos JPG en disco son más rápidos y simples.
# ══════════════════════════════════════════════════════════════════════════════

def guardar_encoding(id_usuario: int, ruta_carpeta: str):
    """
    Registra o actualiza el metadato biométrico del usuario en la BD.
    Guarda la ruta de la carpeta donde están sus imágenes en disco.

    Parámetros:
      id_usuario    — ID del usuario registrado
      ruta_carpeta  — ruta a la carpeta de imágenes
                      Ejemplo: "data_rostros/5_Juan_Garcia"
    """
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO datos_biometricos (id_usuario, encoding, fecha_actualizacion) "
            "VALUES (?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(id_usuario) DO UPDATE SET "
            "encoding = excluded.encoding, "
            "fecha_actualizacion = CURRENT_TIMESTAMP",
            (id_usuario, ruta_carpeta)
        )
    print(f"[BD] Metadato biométrico guardado: ID {id_usuario} → {ruta_carpeta}")


def tiene_biometrico(id_usuario: int) -> bool:
    """
    Retorna True si el usuario tiene rostro registrado.
    CAMBIO: verifica en datos_biometricos Y que la carpeta exista físicamente
    en disco. Evita falsos positivos cuando la carpeta fue eliminada del disco.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT encoding FROM datos_biometricos WHERE id_usuario = ?",
            (id_usuario,)
        ).fetchone()

    if not row:
        return False

    ruta = row["encoding"]
    if not os.path.isdir(ruta):
        print(f"[AVISO] Metadato en BD pero carpeta no encontrada: {ruta}")
        return False

    return True


def obtener_todos_encodings() -> list:
    """
    Retorna lista de (id_usuario, ruta_carpeta) de todos los usuarios activos
    que tienen datos biométricos registrados Y cuya carpeta existe en disco.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT db.id_usuario, db.encoding "
            "FROM datos_biometricos db "
            "JOIN usuarios u ON u.id_usuario = db.id_usuario "
            "WHERE u.estatus = 1"
        ).fetchall()

    resultado = []
    for r in rows:
        ruta = r["encoding"]
        if os.path.isdir(ruta):
            resultado.append((r["id_usuario"], ruta))
        else:
            print(f"[AVISO] Carpeta no encontrada para ID {r['id_usuario']}: {ruta}")

    return resultado


def obtener_encoding(id_usuario: int):
    """Retorna la ruta de carpeta del usuario o None."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT encoding FROM datos_biometricos WHERE id_usuario = ?",
            (id_usuario,)
        ).fetchone()
    return row["encoding"] if row else None


def obtener_carpeta_usuario(id_usuario: int) -> str | None:
    """Retorna la ruta de la carpeta de imágenes del usuario o None."""
    return obtener_encoding(id_usuario)


# ══════════════════════════════════════════════════════════════════════════════
#  ACCESOS  (solo entradas — sin registro de salida)
# ══════════════════════════════════════════════════════════════════════════════

def registrar_entrada(id_usuario: int, metodo: str = "facial") -> int:
    """
    Registra una entrada.
    Si el mismo usuario ya tiene un registro en los últimos 10 segundos,
    se descarta silenciosamente para evitar duplicados por reconocimiento múltiple.
    Pasados los 10 segundos, cualquier nueva pasada genera un registro nuevo.
    Retorna el id_acceso (nuevo o existente).
    """
    hoy        = date.today().isoformat()
    hora_ahora = datetime.now().strftime("%H:%M:%S")

    with get_connection() as conn:
        reciente = conn.execute(
            "SELECT id_acceso FROM accesos "
            "WHERE id_usuario = ? AND fecha = ? "
            "AND hora_entrada >= time('now', '-10 seconds', 'localtime')",
            (id_usuario, hoy)
        ).fetchone()

        if reciente:
            return reciente["id_acceso"]

        cursor = conn.execute(
            "INSERT INTO accesos (id_usuario, fecha, hora_entrada, metodo) "
            "VALUES (?, ?, ?, ?)",
            (id_usuario, hoy, hora_ahora, metodo)
        )
        return cursor.lastrowid


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

# ══════════════════════════════════════════════════════════════════════════════
#  PANEL ADMIN
# ══════════════════════════════════════════════════════════════════════════════

def listar_usuarios() -> list:
    """Retorna todos los usuarios (activos e inactivos) con su rol."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT u.id_usuario, u.nombre, u.apellido_p, u.apellido_m, "
            "u.matricula, u.grado, u.grupo, u.estatus, r.nombre_rol "
            "FROM usuarios u JOIN roles r ON r.id_rol = u.id_rol "
            "ORDER BY u.estatus DESC, u.nombre ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def listar_accesos(limite: int = 50) -> list:
    """Retorna los últimos N accesos con nombre del usuario."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT a.id_acceso, a.fecha, a.hora_entrada, a.metodo, "
            "u.nombre, u.apellido_p, u.matricula "
            "FROM accesos a JOIN usuarios u ON u.id_usuario = a.id_usuario "
            "ORDER BY a.fecha DESC, a.hora_entrada DESC LIMIT ?",
            (limite,)
        ).fetchall()
    return [dict(r) for r in rows]