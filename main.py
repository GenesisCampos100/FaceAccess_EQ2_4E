"""
main.py — Punto de entrada de VisionID
Si no hay admin en la BD, lanza AdminSetup primero.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db_manager import get_connection, ROL_ADMIN


def ya_existe_admin() -> bool:
    """Verificar si ya existe al menos un administrador."""
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as total FROM usuarios WHERE id_rol=? AND estatus=1",
                (ROL_ADMIN,)
            ).fetchone()
        return row["total"] > 0
    except Exception as e:
        print(f"[ERROR] No se pudo verificar admins: {e}")
        return False


if __name__ == "__main__":
    # Verificar si hay admin; si no, lanzar setup
    if not ya_existe_admin():
        print("[INFO] Primer arranque — Configurando administrador...")
        from ui.admin_setup import AdminSetup
        setup = AdminSetup()
        setup.mainloop()
        
        # Si sigue sin haber admin después del setup, salir
        if not ya_existe_admin():
            print("[ERROR] Setup incompleto. No hay administrador en la BD.")
            sys.exit(0)

    # Lanzar aplicación principal
    print("[INFO] Iniciando FaceAccess...")
    from ui.principal import FaceAccess
    app = FaceAccess()
    app.mainloop()