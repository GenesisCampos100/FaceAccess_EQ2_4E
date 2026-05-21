"""
main.py — Punto de entrada de VisionID.
Si no hay admin en la BD, lanza AdminSetup primero.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db_manager import get_connection, ROL_ADMIN

# Modo prueba: mostrar siempre la pantalla de setup inicial.
FORZAR_SETUP_INICIAL = True

def ya_existe_admin() -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as total FROM usuarios WHERE id_rol=? AND estatus=1",
            (ROL_ADMIN,)).fetchone()
    return row["total"] > 0

if __name__ == "__main__":
    if (not ya_existe_admin()):
        from admin import AdminSetup
        setup = AdminSetup()
        try:
            setup.mainloop()
        except KeyboardInterrupt:
            try:
                setup._cerrar()
            except Exception:
                pass
        if not ya_existe_admin():
            sys.exit(0)

    from ReconocimientoFacial import FaceAccess
    app = FaceAccess()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        print('\n[MAIN] Interrupción recibida (KeyboardInterrupt). Cerrando aplicación...')
        try:
            app._cerrar()
        except Exception:
            pass