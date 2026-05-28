"""
ui/constantes.py
Paleta de colores, dimensiones de pantalla y parámetros de visión.
"""

import os
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
SCREEN_W = 600
SCREEN_H = 1042
_S       = 1.0  # sin escala, tamaño de diseño original
_FONT_SCALE = float(os.getenv("FACEACCESS_FONT_SCALE", "1.5"))

def px(n: float) -> int:
    return max(int(n * _S), 1)

def fs(n: float) -> int:
	return max(int(n * _S * _FONT_SCALE), 7)

# ─── Dimensiones de pantalla ──────────────────────────────────────────────────
_DEFAULT_GEOMETRY = os.getenv("FACEACCESS_GEOMETRY", "600x1042")
try:
	APP_W, APP_H = (max(1, int(part)) for part in _DEFAULT_GEOMETRY.lower().split("x", 1))
except ValueError:
	APP_W, APP_H = 600, 1042
APP_GEOMETRY = f"{APP_W}x{APP_H}"

# Para laptop de pruebas: cambiar H_HEADER=58, H_SALUDO=34, APP_GEOMETRY="480x600"
H_HEADER      = 72
H_SALUDO      = 40
H_FOOTER      = max(80, int(APP_H * 0.085))
H_VIDEO       = max(APP_H - H_HEADER - H_SALUDO - H_FOOTER, 1)


def _calc_scale(root=None):
	"""Calcular factor de escala relativo a la resolución objetivo.

	Si se pasa un `root` (Tk/CTk) usa la resolución real de la pantalla,
	si no, asume factor 1.0.
	"""
	try:
		if root is None:
			return 1.0
		sw = root.winfo_screenwidth()
		sh = root.winfo_screenheight()
		if APP_W <= 0 or APP_H <= 0:
			return 1.0
		scale = min(sw / APP_W, sh / APP_H)
		return min(scale, 1.0)
	except Exception:
		return 1.0


def s(value, root=None):
	"""Escala un valor entero (pixeles) según la pantalla."""
	try:
		f = _calc_scale(root)
		return int(max(1, round(value * f)))
	except Exception:
		return int(value)


def sf(point_size, root=None):
	"""Escala un tamaño de fuente (puntos)."""
	try:
		f = _calc_scale(root)
		return max(8, int(round(point_size * f)))
	except Exception:
		return int(point_size)


def sw(percent, root=None):
	"""Devuelve el ancho en píxeles correspondiente a un porcentaje del ancho de app.

	Ej: `sw(0.8, self)` → 80% del ancho objetivo escalado.
	"""
	try:
		f = _calc_scale(root)
		return int(max(1, round(APP_W * percent * f)))
	except Exception:
		return int(APP_W * percent)

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