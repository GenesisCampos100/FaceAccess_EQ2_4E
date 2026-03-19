"""
ReconocimientoFacial.py — Sistema de acceso facial con UI guiada.

Componentes UI:
  · Óvalo guía con colores (gris/amarillo/verde/rojo)
  · Validación de distancia (acércate / aléjate / perfecto)
  · Liveness detection por parpadeo (evita fotos impresas)
  · Barra de progreso de verificación

Flujo completo:
  1. Óvalo gris  → esperando rostro
  2. Óvalo amarillo → rostro detectado, validar distancia
  3. "Acércate" / "Aléjate" si el tamaño no es correcto
  4. Óvalo verde → distancia correcta, detectar parpadeo
  5. Barra de progreso → verificando identidad (3 frames)
  6. Resultado: BIENVENIDO / ACCESO DENEGADO
  7. Pausa 3 segundos → volver al paso 1

Controles:
  ESC → salir
  M   → acceso manual
  L   → login admin
  R   → recargar encodings
"""

import cv2
import os
import json
import threading
import numpy as np
import face_recognition
import imutils
from collections import Counter, deque
from datetime import datetime
from entrenadoRF import cargar_encodings_bd
from db_manager import (
    obtener_usuario_por_id,
    obtener_usuario_por_matricula,
    tiene_entrada_abierta,
    registrar_entrada,
    registrar_salida,
    registrar_intento_fallido,
    guardar_evidencia,
    login,
    puede_registrar,
    nombre_rol,
)

# ─── Configuración ────────────────────────────────────────────────────────────
EVIDENCIAS_DIR   = "evidencias"
TOLERANCIA       = 0.50      # distancia máxima para reconocer
PAUSA_SEG        = 3         # segundos entre reconocimientos
FRAMES_CONFIRM   = 4         # frames para confirmar identidad
ESCALA_DETEC     = 0.25      # reducir frame para detección

# Tamaño del rostro aceptable (porcentaje del alto del frame)
ROSTRO_MIN_RATIO = 0.28      # muy lejos si es menor
ROSTRO_MAX_RATIO = 0.75      # muy cerca si es mayor

# Liveness — parpadeo
UMBRAL_OJO_ABIERTO  = 0.25   # EAR mínimo para considerar ojo abierto
UMBRAL_OJO_CERRADO  = 0.20   # EAR máximo para considerar ojo cerrado
PARPADEOS_REQUERIDOS = 1     # parpadeos necesarios para confirmar vida

# Óvalo guía
OVALO_COLOR_ESPERA   = (120, 120, 120)   # gris
OVALO_COLOR_DETECTADO = (0, 200, 255)   # amarillo
OVALO_COLOR_LISTO    = (0, 255, 0)      # verde
OVALO_COLOR_DENEGADO = (0, 0, 255)      # rojo
OVALO_COLOR_EXITO    = (0, 255, 150)    # verde claro


# ══════════════════════════════════════════════════════════════════════════════
#  LIVENESS DETECTION — EAR (Eye Aspect Ratio)
# ══════════════════════════════════════════════════════════════════════════════

def calcular_ear(ojo: np.ndarray) -> float:
    """
    Calcula el Eye Aspect Ratio (EAR) de un ojo.
    EAR bajo = ojo cerrado, EAR alto = ojo abierto.
    Fórmula: (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    """
    A = np.linalg.norm(ojo[1] - ojo[5])
    B = np.linalg.norm(ojo[2] - ojo[4])
    C = np.linalg.norm(ojo[0] - ojo[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0


def obtener_ear_frame(gray_frame, ubicacion: tuple) -> float | None:
    """
    Obtiene el EAR promedio de ambos ojos para una ubicación de rostro.
    Retorna None si no puede obtener los landmarks.
    """
    top, right, bottom, left = ubicacion
    try:
        landmarks = face_recognition.face_landmarks(
            gray_frame, [(top, right, bottom, left)]
        )
        if not landmarks:
            return None

        lm          = landmarks[0]
        ojo_izq     = np.array(lm.get("left_eye", []))
        ojo_der     = np.array(lm.get("right_eye", []))

        if len(ojo_izq) < 6 or len(ojo_der) < 6:
            return None

        ear_izq = calcular_ear(ojo_izq)
        ear_der = calcular_ear(ojo_der)
        return (ear_izq + ear_der) / 2.0
    except Exception:
        return None


class DetectorParpadeo:
    """Detecta parpadeos reales para liveness detection."""

    def __init__(self, parpadeos_requeridos: int = 1):
        self.parpadeos_requeridos = parpadeos_requeridos
        self.contador_parpadeos  = 0
        self.ojos_cerrados       = False
        self.historial_ear       = deque(maxlen=10)

    def reset(self):
        self.contador_parpadeos = 0
        self.ojos_cerrados      = False
        self.historial_ear.clear()

    def actualizar(self, ear: float) -> bool:
        """
        Actualiza con nuevo valor EAR.
        Retorna True si ya se detectaron los parpadeos requeridos.
        """
        self.historial_ear.append(ear)

        if ear < UMBRAL_OJO_CERRADO:
            self.ojos_cerrados = True
        elif ear > UMBRAL_OJO_ABIERTO and self.ojos_cerrados:
            # Transición cerrado → abierto = 1 parpadeo
            self.contador_parpadeos += 1
            self.ojos_cerrados = False

        return self.contador_parpadeos >= self.parpadeos_requeridos

    @property
    def progreso(self) -> float:
        """Retorna progreso de 0.0 a 1.0."""
        return min(self.contador_parpadeos / self.parpadeos_requeridos, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  ESTADO DEL SISTEMA
# ══════════════════════════════════════════════════════════════════════════════

class EstadoSistema:
    ESPERANDO   = "esperando"
    DETECTADO   = "detectado"
    LIVENESS    = "liveness"
    VERIFICANDO = "verificando"
    RESULTADO   = "resultado"
    PAUSA       = "pausa"

    def __init__(self):
        self.lock              = threading.Lock()
        self.fase              = self.ESPERANDO
        self.corriendo         = True

        # Threading
        self.frame_pendiente   = None
        self.hay_frame_nuevo   = False
        self.resultado_hilo    = None
        self.hay_resultado     = False

        # Reconocimiento
        self.buffer_ids        = []
        self.ultimo_resultado  = None

        # Liveness
        self.detector_parpadeo = DetectorParpadeo(PARPADEOS_REQUERIDOS)

        # Pausa
        self.tiempo_pausa      = None
        self.mensaje_resultado = None

        # UI
        self.ultimo_coords     = None   # coords del rostro para dibujar óvalo
        self.progreso_barra    = 0.0    # 0.0 a 1.0

    def poner_frame(self, frame):
        with self.lock:
            self.frame_pendiente = frame.copy()
            self.hay_frame_nuevo = True

    def tomar_frame(self):
        with self.lock:
            if not self.hay_frame_nuevo:
                return None
            self.hay_frame_nuevo = False
            return self.frame_pendiente

    def poner_resultado_hilo(self, res):
        with self.lock:
            self.resultado_hilo = res
            self.hay_resultado  = True

    def tomar_resultado_hilo(self):
        with self.lock:
            if not self.hay_resultado:
                return None
            self.hay_resultado = False
            return self.resultado_hilo

    def iniciar_pausa(self, mensaje: dict):
        self.fase             = self.PAUSA
        self.tiempo_pausa     = datetime.now()
        self.mensaje_resultado = mensaje
        self.buffer_ids       = []
        self.detector_parpadeo.reset()
        self.progreso_barra   = 0.0

    def verificar_pausa(self) -> bool:
        """Retorna True si la pausa terminó."""
        if self.fase != self.PAUSA:
            return True
        seg = (datetime.now() - self.tiempo_pausa).total_seconds()
        if seg >= PAUSA_SEG:
            self.fase             = self.ESPERANDO
            self.mensaje_resultado = None
            return True
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS VISUALES
# ══════════════════════════════════════════════════════════════════════════════

def guardar_foto_evidencia(frame, prefijo: str) -> str:
    if not os.path.exists(EVIDENCIAS_DIR):
        os.makedirs(EVIDENCIAS_DIR)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ruta = os.path.join(EVIDENCIAS_DIR, f"{prefijo}_{ts}.jpg")
    cv2.imwrite(ruta, frame)
    return ruta


def dibujar_ovalo(frame, color: tuple, grosor: int = 3):
    """Dibuja el óvalo guía centrado en el frame."""
    h, w    = frame.shape[:2]
    cx, cy  = w // 2, h // 2
    rx, ry  = int(w * 0.22), int(h * 0.42)
    cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, color, grosor)


def dibujar_instruccion(frame, texto: str, color: tuple = (255, 255, 255)):
    """Texto de instrucción centrado en la parte superior."""
    h, w = frame.shape[:2]
    tam  = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
    x    = (w - tam[0]) // 2
    # Fondo
    cv2.rectangle(frame, (x - 10, 12), (x + tam[0] + 10, 45), (0, 0, 0), -1)
    cv2.putText(frame, texto, (x, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv2.LINE_AA)


def dibujar_barra_progreso(frame, progreso: float, color: tuple):
    """Barra de progreso en la parte inferior del frame."""
    h, w    = frame.shape[:2]
    margen  = 40
    alto    = 12
    y       = h - 25
    # Fondo de la barra
    cv2.rectangle(frame, (margen, y), (w - margen, y + alto),
                  (50, 50, 50), -1)
    # Progreso
    ancho_lleno = int((w - 2 * margen) * progreso)
    if ancho_lleno > 0:
        cv2.rectangle(frame, (margen, y),
                      (margen + ancho_lleno, y + alto), color, -1)
    # Borde
    cv2.rectangle(frame, (margen, y), (w - margen, y + alto),
                  (180, 180, 180), 1)


def dibujar_resultado(frame, mensaje: dict):
    """Banner de resultado al reconocer."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 115), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    color = mensaje["color"]
    cv2.putText(frame, mensaje["linea3"], (15, h - 82),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(frame, mensaje["linea1"], (15, h - 47),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, mensaje["linea2"], (15, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


def validar_distancia(h_frame: int, alto_rostro: int) -> str:
    """
    Valida si el rostro está a la distancia correcta.
    Retorna: 'ok', 'acercate', 'alejate'
    """
    ratio = alto_rostro / h_frame
    if ratio < ROSTRO_MIN_RATIO:
        return "acercate"
    elif ratio > ROSTRO_MAX_RATIO:
        return "alejate"
    return "ok"


# ══════════════════════════════════════════════════════════════════════════════
#  HILO DE RECONOCIMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def hilo_reconocimiento(estado: EstadoSistema,
                         encodings_ref: list, ids_ref: list):
    while estado.corriendo:
        frame = estado.tomar_frame()
        if frame is None:
            continue

        small = cv2.resize(frame, (0, 0), fx=ESCALA_DETEC, fy=ESCALA_DETEC)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        ubicaciones = face_recognition.face_locations(rgb, model="hog")

        if not ubicaciones:
            estado.poner_resultado_hilo({"tipo": "sin_rostro"})
            continue

        # Escalar coordenadas al frame original
        escala = int(1 / ESCALA_DETEC)
        top, right, bottom, left = ubicaciones[0]
        coords_orig = (top * escala, right * escala,
                       bottom * escala, left * escala)

        # Encoding del rostro
        encodings = face_recognition.face_encodings(rgb, [ubicaciones[0]])
        if not encodings:
            estado.poner_resultado_hilo({"tipo": "sin_rostro"})
            continue

        enc = encodings[0]

        # Comparar contra BD
        if encodings_ref:
            distancias = face_recognition.face_distance(encodings_ref, enc)
            idx        = int(np.argmin(distancias))
            distancia  = float(distancias[idx])
            id_usuario = ids_ref[idx] if distancia <= TOLERANCIA else None
        else:
            id_usuario, distancia = None, 1.0

        estado.poner_resultado_hilo({
            "tipo"      : "rostro",
            "id_usuario": id_usuario,
            "distancia" : distancia,
            "coords"    : coords_orig,
            "frame"     : frame.copy(),
        })


# ══════════════════════════════════════════════════════════════════════════════
#  REGISTRAR ACCESO EN BD
# ══════════════════════════════════════════════════════════════════════════════

def registrar_acceso(resultado: dict, estado: EstadoSistema):
    id_usuario = resultado["id_usuario"]
    frame_cap  = resultado["frame"]
    ahora      = datetime.now()
    dt_str     = ahora.strftime("%d/%m/%Y  %H:%M:%S")

    if id_usuario is not None:
        usuario = obtener_usuario_por_id(id_usuario)
        nombre  = (f"{usuario['nombre']} {usuario['apellido_p']}"
                   if usuario else f"ID {id_usuario}")

        if tiene_entrada_abierta(id_usuario):
            registrar_salida(id_usuario)
            guardar_foto_evidencia(frame_cap, f"salida_{id_usuario}")
            print(f"[BD] SALIDA  — {nombre} | {dt_str}")
            mensaje = {"linea1": nombre, "linea2": dt_str,
                       "linea3": "HASTA LUEGO",
                       "color": (0, 165, 255),
                       "ovalo_color": OVALO_COLOR_EXITO}
        else:
            id_acceso = registrar_entrada(id_usuario, "FACIAL")
            foto      = guardar_foto_evidencia(frame_cap, f"entrada_{id_usuario}")
            guardar_evidencia(id_acceso, foto, "Entrada facial")
            print(f"[BD] ENTRADA — {nombre} | {dt_str}")
            mensaje = {"linea1": nombre, "linea2": dt_str,
                       "linea3": "BIENVENIDO/A",
                       "color": (0, 255, 0),
                       "ovalo_color": OVALO_COLOR_EXITO}
    else:
        guardar_foto_evidencia(frame_cap, "desconocido")
        registrar_intento_fallido("Rostro no reconocido")
        print(f"[BD] DENEGADO | dist: {resultado['distancia']:.3f} | {dt_str}")
        mensaje = {"linea1": "Rostro no registrado", "linea2": dt_str,
                   "linea3": "ACCESO DENEGADO",
                   "color": (0, 0, 255),
                   "ovalo_color": OVALO_COLOR_DENEGADO}

    estado.iniciar_pausa(mensaje)


# ══════════════════════════════════════════════════════════════════════════════
#  ACCESO MANUAL Y LOGIN ADMIN
# ══════════════════════════════════════════════════════════════════════════════

def procesar_acceso_manual(ultimo_frame):
    print("\n" + "="*40)
    print("  ACCESO MANUAL")
    print("="*40)
    matricula = input("Matricula: ").strip()
    usuario   = obtener_usuario_por_matricula(matricula)
    dt_str    = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")

    if not usuario:
        guardar_foto_evidencia(ultimo_frame, "manual_fallido")
        registrar_intento_fallido(
            f"Acceso manual — matricula no encontrada: {matricula}",
            matricula=matricula
        )
        print(f"[BD] Matricula '{matricula}' no encontrada.")
        return

    nombre    = f"{usuario['nombre']} {usuario['apellido_p']}"
    id_acceso = registrar_entrada(usuario["id_usuario"], "MANUAL")
    foto      = guardar_foto_evidencia(ultimo_frame, f"manual_{usuario['id_usuario']}")
    guardar_evidencia(id_acceso, foto, "Entrada manual")
    print(f"[BD] ENTRADA MANUAL — {nombre} | {dt_str}")


def procesar_login_admin(encodings_bd, ids_bd, matricula,
                          contrasenia, resultado) -> dict | None:
    usuario_bd = login(matricula, contrasenia)
    if not usuario_bd:
        registrar_intento_fallido("Credenciales incorrectas", matricula=matricula)
        print("[LOGIN] Credenciales incorrectas.")
        return None

    id_u = resultado.get("id_usuario") if resultado else None
    if id_u is None or resultado.get("distancia", 1.0) > TOLERANCIA:
        registrar_intento_fallido("Rostro no reconocido en login",
                                   matricula=matricula,
                                   id_usuario=usuario_bd["id_usuario"])
        print("[LOGIN] Rostro no reconocido.")
        return None

    if id_u != usuario_bd["id_usuario"]:
        registrar_intento_fallido("Rostro no coincide",
                                   matricula=matricula,
                                   id_usuario=usuario_bd["id_usuario"])
        print("[LOGIN] Rostro no coincide.")
        return None

    if not puede_registrar(usuario_bd["id_rol"]):
        registrar_intento_fallido(
            f"Sin permisos — {nombre_rol(usuario_bd['id_rol'])}",
            matricula=matricula, id_usuario=usuario_bd["id_usuario"]
        )
        print("[LOGIN] Sin permisos.")
        return None

    print(f"[LOGIN] Concedido — {usuario_bd['nombre']}")
    return dict(usuario_bd)


def abrir_interfaz_registro(usuario_sesion: dict):
    print("\n" + "="*40)
    print(f"  SESION : {usuario_sesion['nombre']} {usuario_sesion['apellido_p']}")
    print(f"  ROL    : {usuario_sesion['nombre_rol']}")
    print("="*40)
    # from interfaz_registro import abrir_ventana
    # abrir_ventana(usuario_sesion)


# ══════════════════════════════════════════════════════════════════════════════
#  LOOP PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def iniciar():
    print("[INFO] Cargando encodings desde BD...")
    encodings_bd, ids_bd = cargar_encodings_bd()

    if not encodings_bd:
        print("[ERROR] Sin encodings. Ejecuta capturar_rostro.py.")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    estado = EstadoSistema()

    hilo = threading.Thread(
        target=hilo_reconocimiento,
        args=(estado, encodings_bd, ids_bd),
        daemon=True
    )
    hilo.start()

    modo_login         = False
    credenciales_login = None
    ultimo_frame       = None
    ultimo_resultado   = None
    frame_count        = 0

    print("[INFO] Sistema activo")
    print("       ESC=salir | M=manual | L=login | R=recargar")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count  += 1
        ultimo_frame  = frame.copy()
        h, w          = frame.shape[:2]

        # Enviar frame al hilo de reconocimiento (1 de cada 3)
        if frame_count % 3 == 0 and estado.fase not in (
                EstadoSistema.PAUSA, EstadoSistema.RESULTADO):
            estado.poner_frame(frame)

        # ── FASE: PAUSA ────────────────────────────────────────────────────
        if estado.fase == EstadoSistema.PAUSA:
            estado.verificar_pausa()
            msg = estado.mensaje_resultado
            if msg:
                dibujar_ovalo(frame, msg.get("ovalo_color", OVALO_COLOR_ESPERA), 4)
                dibujar_resultado(frame, msg)
                seg       = (datetime.now() - estado.tiempo_pausa).total_seconds()
                restante  = max(0.0, PAUSA_SEG - seg)
                dibujar_barra_progreso(frame, 1.0 - restante / PAUSA_SEG,
                                       msg["color"])
            cv2.imshow("FaceAccess", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            continue

        # ── Tomar resultado del hilo ───────────────────────────────────────
        resultado = estado.tomar_resultado_hilo()
        if resultado:
            ultimo_resultado = resultado

        # ── Determinar fase según resultado ───────────────────────────────
        sin_rostro = (not ultimo_resultado or
                      ultimo_resultado.get("tipo") == "sin_rostro")

        if sin_rostro:
            estado.fase = EstadoSistema.ESPERANDO
            estado.detector_parpadeo.reset()
            estado.buffer_ids   = []
            estado.progreso_barra = 0.0

        else:
            coords     = ultimo_resultado["coords"]
            top, right, bottom, left = coords
            alto_rostro = bottom - top
            distancia_val = validar_distancia(h, alto_rostro)

            if distancia_val != "ok":
                estado.fase = EstadoSistema.DETECTADO
                estado.detector_parpadeo.reset()
                estado.buffer_ids = []
            else:
                # Distancia correcta — verificar liveness
                if estado.fase in (EstadoSistema.ESPERANDO,
                                    EstadoSistema.DETECTADO):
                    estado.fase = EstadoSistema.LIVENESS

                if estado.fase == EstadoSistema.LIVENESS:
                    gray_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ear      = obtener_ear_frame(gray_rgb, coords)

                    if ear is not None:
                        vivo = estado.detector_parpadeo.actualizar(ear)
                        estado.progreso_barra = (
                            estado.detector_parpadeo.progreso * 0.4
                        )
                        if vivo:
                            estado.fase = EstadoSistema.VERIFICANDO

                if estado.fase == EstadoSistema.VERIFICANDO:
                    estado.buffer_ids.append(
                        ultimo_resultado.get("id_usuario")
                    )
                    estado.progreso_barra = (
                        0.4 + (len(estado.buffer_ids) / FRAMES_CONFIRM) * 0.6
                    )

                    if len(estado.buffer_ids) >= FRAMES_CONFIRM:
                        conteo   = Counter(estado.buffer_ids)
                        id_final = conteo.most_common(1)[0][0]
                        ultimo_resultado["id_usuario"] = id_final

                        if not modo_login:
                            registrar_acceso(ultimo_resultado, estado)
                        else:
                            if credenciales_login:
                                m, c = credenciales_login
                                u    = procesar_login_admin(
                                    encodings_bd, ids_bd, m, c,
                                    ultimo_resultado
                                )
                                if u:
                                    estado.corriendo = False
                                    cap.release()
                                    cv2.destroyAllWindows()
                                    abrir_interfaz_registro(u)
                                    estado.corriendo = True
                                    encodings_bd, ids_bd = cargar_encodings_bd()
                                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                    hilo = threading.Thread(
                                        target=hilo_reconocimiento,
                                        args=(estado, encodings_bd, ids_bd),
                                        daemon=True
                                    )
                                    hilo.start()
                            modo_login         = False
                            credenciales_login = None
                            if estado.fase != EstadoSistema.PAUSA:
                                estado.fase = EstadoSistema.ESPERANDO

        # ── DIBUJAR UI según fase ──────────────────────────────────────────

        if estado.fase == EstadoSistema.ESPERANDO:
            dibujar_ovalo(frame, OVALO_COLOR_ESPERA, 2)
            dibujar_instruccion(frame, "Coloca tu rostro en el ovalo",
                                (200, 200, 200))

        elif estado.fase == EstadoSistema.DETECTADO:
            dibujar_ovalo(frame, OVALO_COLOR_DETECTADO, 3)
            if distancia_val == "acercate":
                dibujar_instruccion(frame, "Acercate a la camara",
                                    (0, 200, 255))
            else:
                dibujar_instruccion(frame, "Alejate un poco",
                                    (0, 200, 255))

        elif estado.fase == EstadoSistema.LIVENESS:
            dibujar_ovalo(frame, OVALO_COLOR_LISTO, 3)
            dibujar_instruccion(frame, "Parpadea para confirmar",
                                (0, 255, 0))
            dibujar_barra_progreso(frame, estado.progreso_barra,
                                   OVALO_COLOR_LISTO)

        elif estado.fase == EstadoSistema.VERIFICANDO:
            dibujar_ovalo(frame, OVALO_COLOR_LISTO, 4)
            dibujar_instruccion(frame, "Verificando identidad...",
                                (0, 255, 150))
            dibujar_barra_progreso(frame, estado.progreso_barra,
                                   (0, 255, 150))

        # Banner login admin
        if modo_login:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 45), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, "LOGIN ADMIN — Parpadea para confirmar",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 200, 0), 1, cv2.LINE_AA)

        cv2.imshow("FaceAccess", frame)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

        elif k == ord("m") or k == ord("M"):
            if estado.fase == EstadoSistema.ESPERANDO:
                procesar_acceso_manual(ultimo_frame)
                estado.iniciar_pausa({
                    "linea1": "Acceso manual registrado",
                    "linea2": datetime.now().strftime("%d/%m/%Y  %H:%M:%S"),
                    "linea3": "ACCESO MANUAL",
                    "color": (255, 165, 0),
                    "ovalo_color": (255, 165, 0)
                })

        elif k == ord("l") or k == ord("L"):
            if estado.fase == EstadoSistema.ESPERANDO and not modo_login:
                print("\n[INFO] LOGIN ADMIN")
                matricula          = input("Matricula : ").strip()
                contrasenia        = input("Contrasena: ").strip()
                credenciales_login = (matricula, contrasenia)
                modo_login         = True
                print("[INFO] Coloca tu rostro y parpadea.")

        elif k == ord("r") or k == ord("R"):
            print("[INFO] Recargando encodings...")
            encodings_bd, ids_bd = cargar_encodings_bd()
            print(f"[OK] {len(encodings_bd)} encoding(s) cargados.")

    estado.corriendo = False
    hilo.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Sistema detenido.")


if __name__ == "__main__":
    iniciar()
