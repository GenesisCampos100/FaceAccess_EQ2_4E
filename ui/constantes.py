"""
ui/constantes.py
Paleta de colores, dimensiones de pantalla y parámetros de visión.
"""

import cv2

# ─── Paleta de colores ────────────────────────────────────────────────────────
C_BG    = "#0F1923"
C_FRAME = "#1A2B3C"
C_FOOT  = "#111E2A"
C_BORDE = "#243447"
C_OK    = "#00D4AA"
C_WARN  = "#F5A623"
C_ERROR = "#E24B4A"
C_TXT   = "#E0EAF4"
C_TXT2  = "#6B8CAE"
C_TXT3  = "#4A6280"
C_ADMIN = "#534AB7"

# ─── Escalado dinámico ────────────────────────────────────────────────────────
# ── PRODUCCIÓN (pantalla 7" real) — descomentar esto y comentar PRUEBAS ──────
# import tkinter as _tk
# _r = _tk.Tk(); _r.withdraw()
# SCREEN_W = _r.winfo_screenwidth()
# SCREEN_H = _r.winfo_screenheight()
# _r.destroy()
# if SCREEN_W > SCREEN_H:
#     SCREEN_W, SCREEN_H = SCREEN_H, SCREEN_W
# _S = min(SCREEN_W / 480, SCREEN_H / 800)

# ── PRUEBAS en laptop — comentar esto en producción ───────────────────────────
SCREEN_W = 480
SCREEN_H = 600
_S       = 1.0  # sin escala, tamaño de diseño original

def px(n: float) -> int:
    return max(int(n * _S), 1)

def fs(n: float) -> int:
    return max(int(n * _S), 7)

# ─── Dimensiones de pantalla ──────────────────────────────────────────────────
H_HEADER     = px(72)
H_SALUDO     = px(40)
H_VIDEO      = SCREEN_H - H_HEADER - H_SALUDO
APP_GEOMETRY = f"{SCREEN_W}x{SCREEN_H}"

# ─── Parámetros de visión ─────────────────────────────────────────────────────
HAAR_CASCADE     = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
ESCALA_DETEC     = 0.5
MIN_VECINOS      = 5
MIN_TAMANO_RELAT = 0.08
UMBRAL_CONFIANZA = 70.0

# ─── Parámetros de flujo ──────────────────────────────────────────────────────
FRAMES_CONFIRM = 2
PAUSA_SEG      = 3
FALLOS_NUMPAD  = 2
EVIDENCIAS_DIR = "evidencias"
FOTOS_CAPTURA  = 30

# ─── Teclado táctil ───────────────────────────────────────────────────────────
KB_APP_W = px(440)
KB_COLS  = 10
KB_PAD   = max(int(2 * _S), 1)
KB_BH    = px(44)
KB_FS    = fs(13)
KB_ACT_H = px(48)