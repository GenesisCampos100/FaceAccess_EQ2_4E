"""
main.py — Punto de entrada de VisionID.
Si no hay admin en la BD, lanza AdminSetup primero.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db_manager import get_connection, ROL_ADMIN

def ya_existe_admin() -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as total FROM usuarios WHERE id_rol=? AND estatus=1",
            (ROL_ADMIN,)).fetchone()
    return row["total"] > 0

if __name__ == "__main__":
    if not ya_existe_admin():
        from admin import AdminSetup
        setup = AdminSetup()
        setup.mainloop()
        if not ya_existe_admin():
            sys.exit(0)

    from ReconocimientoFacial import FaceAccess
    app = FaceAccess()
    app.mainloop()