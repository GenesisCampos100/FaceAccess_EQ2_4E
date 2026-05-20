# VisionID — FaceAccess Modular

## 🏗️ Arquitectura modular

La aplicación sigue un diseño **limpio y modular** con separación clara de responsabilidades:

```
proyecto/
├── main.py                      # Punto de entrada único
│
├── ui/                          # Interfaz gráfica
│   ├── __init__.py
│   ├── principal.py            # FaceAccess (app principal)
│   ├── admin_setup.py          # AdminSetup (setup inicial)
│   ├── teclado.py              # TecladoVirtual
│   ├── constantes.py           # Colores, dimensiones, parámetros
│   └── ...
│
├── core/                        # Lógica de bajo nivel
│   ├── __init__.py
│   ├── camara.py               # CamaraManager (abstracción de cámara)
│   ├── reconocimiento.py       # buscar() - predicción LBPH
│   ├── captura.py              # Captura de rostros
│   └── ...
│
├── database/                    # Acceso a datos
│   ├── __init__.py
│   ├── db_manager.py           # Todas las operaciones SQLite3
│   ├── control_acceso.db       # Base de datos
│   ├── data_rostros/           # Imágenes de rostros guardadas
│   └── ...
│
├── models/                      # Modelos de ML
│   ├── __init__.py
│   ├── entrenadoRF.py          # LBPH training y loading
│   ├── modelo_lbph.xml         # Modelo entrenado
│   └── ...
│
├── assets/                      # Recursos estáticos
│   └── icono.png
│
└── README.md (este archivo)
```

## 📦 Módulos

### `ui/` — Interfaz gráfica
- **`principal.py`**: Clase `FaceAccess` - aplicación principal con:
  - Reconocimiento facial en tiempo real
  - Entrada manual con teclado virtual
  - Modos: acceso, registro, login, captura
  - Manejo de threads para reconocimiento asincrónico

- **`admin_setup.py`**: Clase `AdminSetup` - asistente de configuración:
  - Ejecuta UNA SOLA VEZ en primer arranque
  - Crea usuario administrador inicial
  - Captura 30 imágenes del rostro del admin
  - Entrena modelo LBPH

- **`teclado.py`**: Clase `TecladoVirtual` - teclado táctil:
  - Interfaz numérica + alfanumérica
  - Optimizada para pantalla táctil 7"
  - Soporte para contraseña (mostrar/ocultar)

- **`constantes.py`**: Configuración centralizada:
  - Colores (`C_BG`, `C_OK`, `C_ERROR`, etc.)
  - Dimensiones UI (`H_HEADER`, `H_SALUDO`, `APP_GEOMETRY`, etc.)
  - Parámetros Haar Cascade (`HAAR_CASCADE`, `ESCALA_DETEC`, `MIN_VECINOS`, etc.)
  - Parámetros LBPH (`UMBRAL_CONFIANZA`, `FRAMES_CONFIRM`, etc.)

### `core/` — Lógica de reconocimiento
- **`camara.py`**: Clase `CamaraManager` - abstracción de cámara:
  - Soporta Raspberry Pi CSI (`picamera2`) o USB webcam (`OpenCV`)
  - API única: `iniciar()`, `leer()`, `liberar()`
  - Thread-safe con locks compartidos

- **`reconocimiento.py`**: Funciones puras de reconocimiento:
  - `cargar_encodings()` - carga modelo LBPH desde disco
  - `buscar(rostro_gray, recognizer)` - predicción + preprocesamiento:
    - CLAHE (adaptative histogram equalization)
    - Bilateral filter (denoise)
    - Normalización de rango
    - Comparación distancia chi-squared vs threshold
  - **Sin dependencias UI ni DB** → reutilizable

- **`captura.py`**: Captura de rostros:
  - `seleccionar_usuario()` - prompt interactivo
  - `capturar(usuario)` - loop de captura 30 imágenes
  - Normalización automática (200x200, escala gris)
  - Re-entrena modelo al finalizar

### `database/` — Acceso a datos
- **`db_manager.py`**: Módulo centralizado de BD:
  - Operaciones: login, registrar_usuario, guardar_encoding, etc.
  - Query sanitizadas (protección SQL injection)
  - WAL journal mode para sqlite3
  - Constantes: `DATA_DIR`, `ROL_*`, etc.

- **`control_acceso.db`**: Base de datos SQLite3 con tablas:
  - `usuarios` — usuarios y autenticación
  - `roles` — roles (admin, personal, estudiante)
  - `datos_biometricos` — rutas de carpetas de rostros
  - `accesos` — log de accesos
  - `intentos_fallidos` — log de intentos fallidos
  - `evidencias` — fotos de comprobante

- **`data_rostros/`**: Carpetas por usuario:
  - `1_GENESIS_CAMPOS/` → rostro_000.jpg, rostro_001.jpg, ...
  - `2_LUZ_CAMPOS/` → ...
  - Imágenes en escala gris 200x200 (formato LBPH)

### `models/` — Modelos de ML
- **`entrenadoRF.py`**: Entrenamiento LBPH:
  - `cargar_dataset()` - lee todas imágenes de data_rostros/
  - `entrenar()` - entrena cv2.face.LBPHFaceRecognizer
  - `cargar_modelo_lbph()` - carga modelo desde disco
  - Parámetros: radius=1, neighbors=8, grid_x=8, grid_y=8

- **`modelo_lbph.xml`**: Modelo serializado (OpenCV)

## 🚀 Flujo de ejecución

```
main.py
 ├─ ya_existe_admin()
 │  └─ Si no existe: ui.admin_setup.AdminSetup()
 │     ├─ Formulario datos admin
 │     ├─ core.camara.CamaraManager() → captura 30 rostros
 │     ├─ database.db_manager.registrar_usuario()
 │     └─ models.entrenadoRF.entrenar()
 │
 └─ ui.principal.FaceAccess()
    ├─ core.camara.CamaraManager()
    ├─ core.reconocimiento.cargar_encodings()
    ├─ cv2.CascadeClassifier (Haar)
    └─ Loop principal:
       ├─ Captura frame
       ├─ Detección Haar
       ├─ Predicción LBPH (thread asincrónico)
       ├─ Validación identidad
       └─ Registra acceso en BD
```

## 🔧 Importes modulares

Desde cualquier lugar en el proyecto:

```python
# Interfaz
from ui.principal import FaceAccess
from ui.admin_setup import AdminSetup
from ui.constantes import C_OK, APP_GEOMETRY

# Lógica
from core.camara import CamaraManager
from core.reconocimiento import buscar, cargar_encodings
from core.captura import capturar

# Base de datos
from database.db_manager import login, registrar_usuario, registrar_entrada

# Modelos
from models.entrenadoRF import cargar_modelo_lbph, entrenar
```

O con imports del paquete (via `__init__.py`):

```python
from ui import FaceAccess, AdminSetup
from core import CamaraManager, cargar_encodings
from database import get_connection, login
from models import cargar_modelo_lbph
```

## 📋 Configuración

Toda la configuración está centralizada en `ui/constantes.py`:

```python
# Colores (tema dark)
C_BG = "#0A0E27"           # fondo
C_FRAME = "#1A1F3A"        # marcos
C_OK = "#00D4AA"           # verde éxito
C_ERROR = "#F54E3F"        # rojo error
C_WARN = "#F5A623"         # naranja advertencia

# Dimensiones
APP_GEOMETRY = "600x1024"  # pantalla 7" vertical
H_HEADER = 58              # altura header
H_SALUDO = 34              # altura saludo
H_VIDEO = 600              # altura video

# Parámetros Haar
HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
ESCALA_DETEC = 0.5         # reducción para detección rápida
MIN_VECINOS = 5            # neighbors para Haar
MIN_TAMANO_RELAT = 0.15    # tamaño mínimo relativo

# Parámetros LBPH
UMBRAL_CONFIANZA = 70.0    # distancia máxima aceptada
FRAMES_CONFIRM = 3         # frames consecutivos para confirmar
```

## 🧪 Testing

Para ejecutar módulos individuales:

```bash
# Setup inicial
python -m ui.admin_setup

# Captura de rostros
python -m core.captura

# Entrenar modelo
python -m models.entrenadoRF

# App principal
python main.py
```

## 🛠️ Desarrollo futuro

- [ ] Agregar logging centralizado
- [ ] Añadir validación de movimiento anti-spoofing
- [ ] Soporte para múltiples métodos de autenticación
- [ ] Dashboard de estadísticas de acceso
- [ ] Exportar reportes
- [ ] API REST para integración externa

## 📄 Licencia

Proyecto Integrador — Ingeniería de Software, 4to Semestre
