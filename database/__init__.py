"""
database/ — Módulo de acceso a datos (SQLite3)
"""
from database.db_manager import (
    get_connection,
    login,
    registrar_usuario,
    obtener_usuario_por_id,
    obtener_usuario_por_matricula,
    registrar_entrada,
    registrar_intento_fallido,
    guardar_evidencia,
    guardar_encoding,
    DATA_DIR,
    ROL_ADMIN,
    ROL_PERSONAL_AUTORIZADO,
    ROL_PERSONAL_ESCOLAR,
    ROL_ALUMNO,
)

__all__ = [
    "get_connection",
    "login",
    "registrar_usuario",
    "obtener_usuario_por_id",
    "obtener_usuario_por_matricula",
    "registrar_entrada",
    "registrar_intento_fallido",
    "guardar_evidencia",
    "guardar_encoding",
    "DATA_DIR",
    "ROL_ADMIN",
    "ROL_PERSONAL_AUTORIZADO",
    "ROL_PERSONAL_ESCOLAR",
    "ROL_ALUMNO",
]

