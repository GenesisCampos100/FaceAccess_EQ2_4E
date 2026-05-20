"""
main.py
Punto de entrada del sistema VisionID.
Solo importa y arranca FaceAccess — no contiene lógica.
"""

from ReconocimientoFacial import FaceAccess

if __name__ == "__main__":
    app = FaceAccess()
    app.mainloop()
