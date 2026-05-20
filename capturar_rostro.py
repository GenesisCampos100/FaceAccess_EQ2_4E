"""
capturar_rostro.py — ALIAS MODULAR (uso: python capturar_rostro.py)
Este archivo redirige a la versión modular en core/captura.py

⚠️  DEPRECADO: Usa 'python -m core.captura' en su lugar.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("[ADVERTENCIA] capturar_rostro.py está deprecated.")
    print("[INFO] Ejecutando captura desde módulo core...")
    
    from core.captura import seleccionar_usuario, capturar
    u = seleccionar_usuario()
    if u:
        capturar(u)
