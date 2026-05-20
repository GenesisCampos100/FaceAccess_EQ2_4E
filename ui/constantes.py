"""
ui/constantes.py
Paleta de colores, dimensiones de pantalla y parámetros de visión.
Importar desde aquí en todos los módulos — nunca definir colores en otro lugar.
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

# ─── Dimensiones de pantalla ──────────────────────────────────────────────────
# Para laptop de pruebas: cambiar H_HEADER=58, H_SALUDO=34, APP_GEOMETRY="480x600"
H_HEADER      = 72
H_SALUDO      = 40
H_VIDEO       = 800 - H_HEADER - H_SALUDO
APP_GEOMETRY  = "380x700"   # usar "480x600" en laptop

# ─── Parámetros de visión ─────────────────────────────────────────────────────
HAAR_CASCADE      = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
ESCALA_DETEC      = 0.5
MIN_VECINOS       = 5
MIN_TAMANO_RELAT  = 0.12
UMBRAL_CONFIANZA  = 70.0

# ─── Parámetros de flujo ──────────────────────────────────────────────────────
FRAMES_CONFIRM  = 2
PAUSA_SEG       = 3
FALLOS_NUMPAD   = 2
EVIDENCIAS_DIR  = "evidencias"
FOTOS_CAPTURA   = 30

# ─── Teclado táctil ───────────────────────────────────────────────────────────
KB_APP_W   = 480
KB_COLS    = 11
KB_PAD     = 4
KB_BH      = 52     # usar 38 en laptop
KB_FS      = 16
KB_ACT_H   = 50     # usar 38 en laptop
