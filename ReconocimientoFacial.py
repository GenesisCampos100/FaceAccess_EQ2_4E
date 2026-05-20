"""
ReconocimientoFacial.py — ALIAS MODULAR (uso: python ReconocimientoFacial.py)
Este archivo redirige a la versión modular en ui/principal.py

⚠️  DEPRECADO: Usa 'python main.py' en su lugar.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("[ADVERTENCIA] ReconocimientoFacial.py está deprecated.")
    print("[INFO] Ejecutando versión modular desde main.py...")
    
    from ui.principal import FaceAccess
    app = FaceAccess()
    app.mainloop()
