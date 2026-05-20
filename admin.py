"""
admin.py — ALIAS MODULAR (uso: python admin.py)
Este archivo redirige a la versión modular en ui/admin_setup.py

⚠️  DEPRECADO: Usa 'python main.py' en su lugar.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("[ADVERTENCIA] admin.py está deprecated.")
    print("[INFO] Ejecutando setup desde main.py...")
    
    from ui.admin_setup import AdminSetup
    setup = AdminSetup()
    setup.mainloop()
