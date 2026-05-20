"""
core/ — Módulo de lógica principal y captura
"""
from core.reconocimiento import cargar_encodings, buscar
from core.camara import CamaraManager
from core.captura import capturar, seleccionar_usuario

__all__ = ["cargar_encodings", "buscar", "CamaraManager", "capturar", "seleccionar_usuario"]

