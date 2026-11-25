# -*- coding: utf-8 -*-
"""
AUD-05 v5.22 — CANDIDATA DE VISIÓN EXTERNA (NO CERTIFICADA PARA VUELO)
Documento anexo perteneciente al Trabajo de Fin de Grado: "Vuelo autónomo
y estable de dron en interiores mediante Pixhawk y visión artificial con
marcadores ARUCO".
Autora: Manuela Posada Londoño.
Grado en Ingeniería en Electrónica Industrial y Automática (UVa).

Implementa la corrección rigurosa de todas las incidencias detectadas en la
auditoría estática e integración:
- Eliminación y prevención de espacios no separables (\xa0).
- Propagación determinista y completa del motivo de rechazo visual/óptico.
- Política fail-closed estricta en verificación de rearme ante vehículo ausente.
- Control defensivo de excepciones en llamadas a OpenCV (cvtColor, detectMarkers, solvePnP).
- Sincronización coherente de estado, velocidad segura y órdenes tras fallo o emergencia.
- Validación formal de dimensiones y tipos antes de estimación de pose.
"""

# --- PARCHE DE COMPATIBILIDAD PARA PYTHON 3.10+ (DRONEKIT) ---
import collections
import collections.abc
collections.MutableMapping = collections.abc.MutableMapping
# -------------------------------------------------------------

import sys
import os
import tempfile
import time
from enum import Enum
import cv2
import numpy as np

# Importación defensiva de librerías de vuelo (permite ejecutar --test sin hardware ni enlace)
try:
    from dronekit import connect, VehicleMode
    from pymavlink import mavutil
    DRONEKIT_DISPONIBLE = True
except ImportError:
    DRONEKIT_DISPONIBLE = False

def crear_detector_aruco():
    """Crea detector compatible con OpenCV antiguo y nuevo; falla explícitamente si falta ArUco."""
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("Esta instalación de OpenCV no incluye cv2.aruco; instale opencv-contrib-python.")
    diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "DetectorParameters"):
        parametros = cv2.aruco.DetectorParameters()
    elif hasattr(cv2.aruco, "DetectorParameters_create"):
        parametros = cv2.aruco.DetectorParameters_create()
    else:
        raise RuntimeError("API DetectorParameters de OpenCV no disponible.")
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(diccionario, parametros)
    # OpenCV 4.6 y anteriores: adaptador de la API funcional antigua.
    class DetectorCompat:
        def detectMarkers(self, imagen_gris):
            return cv2.aruco.detectMarkers(imagen_gris, diccionario, parameters=parametros)
    return DetectorCompat()


# ==========================================
# 1. CONSTANTES DE SEGURIDAD Y CONTROL
# ==========================================
DISTANCIA_OBJETIVO_CM = 80.0
MAX_VELOCIDAD = 0.30             # Límite máximo de velocidad longitudinal (m/s)
Kp_z = 0.015                     # Ganancia proporcional longitudinal ((m/s)/cm)
LADO_REAL_CM = 35.8              # Longitud real medida del marcador ArUco (cm)
RESOLUCION_CALIBRADA = (640, 480)  # (ancho, alto) exigido por la calibración óptica
EXIGIR_NPZ_DESDE_DISCO = True      # True: bloquea control autónomo si falta el archivo .npz

# Parámetros ópticos calibrados de referencia estricta para la RunCam Nano 2 (640x480)
MTX_RUNCAM_CALIBRADA = np.array([
    [315.73111283,    0.0,        335.28098147],
    [   0.0,        321.41073403, 276.77736896],
    [   0.0,          0.0,          1.0       ]
], dtype=np.float64)
DIST_RUNCAM_CALIBRADA = np.array(
    [[-0.27058807, 0.04556817, -0.02216294, -0.00290499, -0.00332127]],
    dtype=np.float64
)

ID_ARUCO_OBJETIVO = 0  # ID fijo autorizado; no se acepta el primer marcador arbitrario.

# --- PARÁMETROS DE LA MÁQUINA DE ESTADOS ---
FRAMES_VALIDACION_REQ = 10         # Fotogramas consecutivos válidos para validar visión
TIMEOUT_EMERGENCIA_S = 0.5         # Segundos máximos sin visión o bloqueo en CONTROL_ACTIVO
MAX_DT_VALIDACION_S = 0.25         # Intervalo máximo admisible entre frames en VALIDACION
EPSILON_TIEMPO_S = 1e-6            # Tolerancia numérica IEEE 754
MAX_DT_CICLO_S = TIMEOUT_EMERGENCIA_S  # Pausa de ciclo inadmisible: enclavar emergencia

# MAVLink SET_POSITION_TARGET_LOCAL_NED_TYPEMASK para controlar SOLO Vx/Vy.
# Ignora posición (bits 0-2), Vz (bit 5), aceleración XYZ (bits 6-8),
# yaw (bit 10) y yaw-rate (bit 11). FORCE_SET (bit 9) queda desactivado.
MAVLINK_TYPE_MASK_VELOCIDAD_XY = 0b0000110111100111
MAVLINK_FRAME_VELOCIDAD = "MAV_FRAME_BODY_OFFSET_NED"

# Geometría y frecuencia de la visión externa. El rumbo y los offsets se leen del entorno.
ALTURA_CENTRO_ARUCO_M = 1.5
FRECUENCIA_VISION_MIN_HZ = 4.0
FRECUENCIA_VISION_OBJETIVO_HZ = 30.0
TIMEOUT_POSE_EXTERNA_S = 0.25
EKF_EXTERNAL_NAV_VALUE = 6


# ==========================================
# 2. FUNCIONES AUXILIARES DE VALIDACIÓN DEFENSIVA
# ==========================================
def validar_escalar_real_finito(valor, nombre_parametro):
    """Valida estrictamente que un valor sea un escalar real, finito y no booleano."""
    if isinstance(valor, (bool, np.bool_)):
        raise ValueError(f"Parámetro '{nombre_parametro}': no se admite tipo booleano.")
    if not isinstance(valor, (int, float, np.integer, np.floating)):
        raise ValueError(f"Parámetro '{nombre_parametro}': se requiere un tipo numérico real.")
    
    resultado = float(valor)
    if not np.isfinite(resultado):
        raise ValueError(f"Parámetro '{nombre_parametro}': el valor debe ser finito (recibido {valor}).")
    return resultado


def seleccionar_marcador_objetivo(esquinas, ids, id_objetivo):
    """
    Selecciona siempre el mismo ID objetivo en cualquier camino de control.
    Retorna (esquinas_2d, indice, motivo).
    """
    if esquinas is None or ids is None or len(ids) == 0:
        return None, None, "ARUCO_NO_DETECTADO"
    try:
        ids_planos = np.asarray(ids).reshape(-1)
    except Exception as e:
        return None, None, f"ARUCO_IDS_INVALIDOS ({type(e).__name__}: {e})"

    if id_objetivo is None:
        return None, None, "ID_OBJETIVO_NO_CONFIGURADO"
    else:
        try:
            id_objetivo = int(id_objetivo)
        except (TypeError, ValueError, OverflowError):
            return None, None, "ID_OBJETIVO_INVALIDO"
        coincidencias = np.flatnonzero(ids_planos == id_objetivo)
        if coincidencias.size == 0:
            return (
                None,
                None,
                f"ID_OBJETIVO_NO_DETECTADO ({id_objetivo})"
            )
        indice = int(coincidencias[0])
    
    if indice >= len(esquinas) or esquinas[indice] is None:
        return None, None, "ARUCO_ESQUINAS_INCONSISTENTES"
        
    return esquinas[indice][0], indice, "OK"


def verificar_condiciones_rearme(vehiculo, calibracion_autorizada_ok, modo_simulacion=False):
    """
    Verifica de manera conservadora las condiciones externas de seguridad antes de autorizar el rearme.
    Aplica principio fail-closed estricto si el vehículo es None y no existe modo simulación explícito.
    """
    if vehiculo is None:
        if modo_simulacion:
            return {
                "enlace_verificado": True,
                "vehiculo_desarmado": True,
                "modo_verificado": True,
                "calibracion_autorizada": bool(calibracion_autorizada_ok),
                "autorizacion_operador": False,
                "transporte_verificado": False,
            }
        else:
            return {
                "enlace_verificado": False,
                "vehiculo_desarmado": False,
                "modo_verificado": False,
                "calibracion_autorizada": False,
                "autorizacion_operador": False,
                "transporte_verificado": False,
            }
    try:
        modo = vehiculo.mode.name if vehiculo.mode is not None else None
        heartbeat_reciente = (
            vehiculo.last_heartbeat is not None
            and float(vehiculo.last_heartbeat) < 2.0
        )
        return {
            "enlace_verificado": heartbeat_reciente,
            "vehiculo_desarmado": vehiculo.armed is False,
            "modo_verificado": modo is not None and modo != "UNKNOWN",
            "calibracion_autorizada": bool(calibracion_autorizada_ok),
            "autorizacion_operador": False,
            "transporte_verificado": False,
        }
    except Exception:
        return {
            "enlace_verificado": False,
            "vehiculo_desarmado": False,
            "modo_verificado": False,
            "calibracion_autorizada": False,
            "autorizacion_operador": False,
            "transporte_verificado": False,
        }


# ==========================================
# 3. MÁQUINA DE ESTADOS (FSM) Y ENUMS
# ==========================================
class EstadoVuelo(Enum):
    BUSQUEDA = 1        # Ascenso manual / espera inicial sin referencia previa
    VALIDACION = 2      # Marcador detectado, exigiendo 10 fotogramas consecutivos
    CONTROL_ACTIVO = 3  # Referencia visual validada (10 frames consecutivos superados)
    EMERGENCIA = 4      # Failsafe por pérdida de visión, bloqueo dt, cámara, óptica o transporte


class FSM_Seguridad:
    def __init__(self, frames_req=FRAMES_VALIDACION_REQ, timeout_s=TIMEOUT_EMERGENCIA_S,
                 max_dt_val_s=MAX_DT_VALIDACION_S):
        f_req = validar_escalar_real_finito(frames_req, "frames_req")
        t_out = validar_escalar_real_finito(timeout_s, "timeout_s")
        dt_v = validar_escalar_real_finito(max_dt_val_s, "max_dt_val_s")

        if f_req <= 0 or not float(f_req).is_integer():
            raise ValueError("frames_req debe ser un entero positivo.")
        if t_out <= 0.0:
            raise ValueError("timeout_s debe ser estrictamente positivo.")
        if dt_v <= 0.0:
            raise ValueError("max_dt_val_s debe ser estrictamente positivo.")

        self.estado = EstadoVuelo.BUSQUEDA
        self.frames_consecutivos = 0
        self.timeout_s = t_out
        self.frames_req = int(f_req)
        self.max_dt_val_s = dt_v
        self.ultimo_tiempo_visto = None
        self.tiempo_ciego = 0.0
        self.en_emergencia_latched = False
        self.en_fallo_transporte_latched = False  # Enclavamiento crítico de bus MAVLink

    @property
    def referencia_visual_validada(self):
        return self.estado == EstadoVuelo.CONTROL_ACTIVO and not self.en_fallo_transporte_latched and not self.en_emergencia_latched

    def reiniciar_a_busqueda(self):
        """Rearme operativo por cambio de modo. Preserva enclavamientos críticos."""
        self.frames_consecutivos = 0
        self.ultimo_tiempo_visto = None
        self.tiempo_ciego = 0.0
        if self.en_fallo_transporte_latched or self.en_emergencia_latched:
            self.estado = EstadoVuelo.EMERGENCIA
        else:
            self.estado = EstadoVuelo.BUSQUEDA
            self.en_emergencia_latched = False

    def rearmar_sistema_supervisado(self, *, autorizacion_operador, enlace_verificado, vehiculo_desarmado, modo_verificado, calibracion_autorizada, transporte_verificado):
        """
        Libera el enclavamiento únicamente si todas las condiciones de rearme han sido comprobadas.
        """
        condiciones = (
            autorizacion_operador,
            enlace_verificado,
            vehiculo_desarmado,
            modo_verificado,
            calibracion_autorizada,
            transporte_verificado,
        )
        # Rechazar valores truthy no booleanos (p. ej. 1, objetos o strings).
        if not all(valor is True for valor in condiciones):
            return False
        # Un fallo del transporte invalida la confianza en el canal de mando.
        # No se permite borrar ese enclavamiento desde el mismo proceso con
        # simples booleanos: requiere cerrar el proceso, reconectar y verificar
        # el enlace mediante procedimiento externo supervisado.
        if self.en_fallo_transporte_latched:
            return False
        self.en_fallo_transporte_latched = False
        self.en_emergencia_latched = False
        self.estado = EstadoVuelo.BUSQUEDA
        self.frames_consecutivos = 0
        self.ultimo_tiempo_visto = None
        self.tiempo_ciego = 0.0
        return True

    def forzar_emergencia(self, transporte_roto=False):
        self.estado = EstadoVuelo.EMERGENCIA
        self.frames_consecutivos = 0
        self.en_emergencia_latched = True
        if transporte_roto:
            self.en_fallo_transporte_latched = True
        return self.estado

    def actualizar(self, vision_valida, tiempo_actual=None, navegacion_ekf_valida=True):
        if self.en_fallo_transporte_latched or self.en_emergencia_latched:
            self.estado = EstadoVuelo.EMERGENCIA
            return self.estado

        if tiempo_actual is None:
            tiempo_actual = time.monotonic()

        dt_previo = (
            max(0.0, tiempo_actual - self.ultimo_tiempo_visto)
            if self.ultimo_tiempo_visto is not None
            else 0.0
        )

        if (
            self.estado == EstadoVuelo.CONTROL_ACTIVO
            and self.ultimo_tiempo_visto is not None
            and dt_previo + EPSILON_TIEMPO_S >= self.timeout_s
        ):
            # Watchdog de ciclo: una pausa larga es fallo aunque el frame nuevo sea válido.
            self.estado = EstadoVuelo.EMERGENCIA
            self.frames_consecutivos = 0
            self.tiempo_ciego = dt_previo
            self.en_emergencia_latched = True
            return self.estado

        if self.estado == EstadoVuelo.CONTROL_ACTIVO and not vision_valida:
            # Pérdida de visión durante control activo: no seguir actuando con una pose antigua.
            self.estado = EstadoVuelo.EMERGENCIA
            self.frames_consecutivos = 0
            self.tiempo_ciego = dt_previo
            self.en_emergencia_latched = True
            return self.estado

        if vision_valida:
            self.ultimo_tiempo_visto = tiempo_actual
            self.tiempo_ciego = 0.0

            if self.estado in [EstadoVuelo.BUSQUEDA, EstadoVuelo.EMERGENCIA]:
                self.estado = EstadoVuelo.VALIDACION
                self.frames_consecutivos = 1
            elif self.estado == EstadoVuelo.VALIDACION:
                if dt_previo > self.max_dt_val_s + EPSILON_TIEMPO_S:
                    self.frames_consecutivos = 1
                else:
                    self.frames_consecutivos += 1
                if self.frames_consecutivos >= self.frames_req:
                    self.frames_consecutivos = self.frames_req
                    if navegacion_ekf_valida is True:
                        self.estado = EstadoVuelo.CONTROL_ACTIVO
                        self.en_emergencia_latched = False
            elif self.estado == EstadoVuelo.CONTROL_ACTIVO:
                self.frames_consecutivos = min(self.frames_req, self.frames_consecutivos + 1)
        else:
            self.frames_consecutivos = 0
            self.tiempo_ciego = dt_previo if self.ultimo_tiempo_visto is not None else 0.0
            if self.estado == EstadoVuelo.VALIDACION:
                self.estado = EstadoVuelo.BUSQUEDA

        return self.estado


# ==========================================
# 4. VALIDADOR VISUAL (RECHAZO DE OUTLIERS)
# ==========================================
class ValidadorVisual:
    def __init__(self, limite_mre=8.0, z_min=20.0, z_max=250.0, salto_max=40.0):
        self.limite_mre = validar_escalar_real_finito(limite_mre, "limite_mre")
        self.z_min = validar_escalar_real_finito(z_min, "z_min")
        self.z_max = validar_escalar_real_finito(z_max, "z_max")
        self.salto_max = validar_escalar_real_finito(salto_max, "salto_max")

        if self.limite_mre <= 0.0:
            raise ValueError("limite_mre debe ser estrictamente positivo.")
        if not (0.0 <= self.z_min < self.z_max):
            raise ValueError("Se requiere 0 <= z_min < z_max.")
        if self.salto_max <= 0.0:
            raise ValueError("salto_max debe ser estrictamente positivo.")

    def auditar(self, z_medida, z_anterior, puntos_3d, esquinas_2d, rvec, tvec, mtx, dist, estado_vuelo):
        try:
            z_val = validar_escalar_real_finito(z_medida, "z_medida_auditar")
        except ValueError as e:
            return False, f"Rechazo Valor Inválido: {e}"

        z_ant_val = None
        if z_anterior is not None:
            try:
                z_ant_val = validar_escalar_real_finito(z_anterior, "z_anterior")
            except ValueError:
                pass

        try:
            pts_proyectados, _ = cv2.projectPoints(puntos_3d, rvec, tvec, mtx, dist)
            pts_reales = esquinas_2d.reshape(-1, 2).astype(np.float32)
            pts_proy = pts_proyectados.reshape(-1, 2).astype(np.float32)
        except Exception as e:
            return False, f"Rechazo Proyección OpenCV ({type(e).__name__}: {e})"

        errores_esquinas = np.linalg.norm(pts_reales - pts_proy, axis=1)
        mre = float(np.mean(errores_esquinas))

        if mre > self.limite_mre:
            return False, f"Rechazo MRE: {mre:.1f}px > {self.limite_mre}px"

        if not (self.z_min <= z_val <= self.z_max):
            return False, f"Rechazo Dimensional: {z_val:.1f}cm"

        if estado_vuelo in [EstadoVuelo.VALIDACION, EstadoVuelo.CONTROL_ACTIVO] and z_ant_val is not None:
            salto_espacial = abs(z_val - z_ant_val)
            if salto_espacial > self.salto_max:
                return False, f"Rechazo Cinemático: Salto de {salto_espacial:.1f}cm"

        return True, f"Validado (MRE: {mre:.1f}px)"


# ==========================================
# 5. FILTRO DE KALMAN CINEMÁTICO (FORMA DE JOSEPH)
# ==========================================
class FiltroKalmanCinematico:
    def __init__(self, dt_inicial, var_pos, var_vel, var_medida, pos_inicial):
        dt = validar_escalar_real_finito(dt_inicial, "dt_inicial")
        vp = validar_escalar_real_finito(var_pos, "var_pos")
        vv = validar_escalar_real_finito(var_vel, "var_vel")
        vm = validar_escalar_real_finito(var_medida, "var_medida")
        p0 = validar_escalar_real_finito(pos_inicial, "pos_inicial")

        if dt <= 0.0:
            raise ValueError("dt_inicial debe ser positivo.")
        if vp < 0.0 or vv < 0.0:
            raise ValueError("Las varianzas de proceso no pueden ser negativas.")
        if vm <= 0.0:
            raise ValueError("var_medida debe ser estrictamente positiva.")
        if p0 <= 0.0:
            raise ValueError("pos_inicial debe ser estrictamente positiva (profundidad).")

        self.dt = dt
        self.var_pos_base = vp
        self.var_vel_base = vv
        self.X = np.array([[p0], [0.0]], dtype=np.float64)
        self.F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        self.Q = np.diag([vp * dt, vv * dt])
        self.R = np.array([[vm]], dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64)
        self.I = np.eye(2, dtype=np.float64)

    def reiniciar(self, pos_inicial):
        p0 = validar_escalar_real_finito(pos_inicial, "pos_inicial_reiniciar")
        if p0 <= 0.0:
            raise ValueError("pos_inicial en reinicio debe ser positiva.")
        self.X = np.array([[p0], [0.0]], dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64)
        return self.X[0, 0]

    def _actualizar_modelo_temporal(self, dt_medidas):
        dt = validar_escalar_real_finito(dt_medidas, "dt_medidas")
        if dt <= 0.0:
            raise ValueError("dt_medidas debe ser estrictamente positivo.")
        if dt >= MAX_DT_CICLO_S:
            raise ValueError(
                f"dt_medidas={dt:.3f}s supera el máximo seguro "
                f"de {MAX_DT_CICLO_S:.3f}s; debe gestionarse como fallo de ciclo."
            )
        self.dt = dt
        self.F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        self.Q = np.diag([self.var_pos_base * dt, self.var_vel_base * dt])

    def actualizar(self, medida, dt_medidas=None):
        z_val = validar_escalar_real_finito(medida, "medida_kalman")
        if z_val <= 0.0:
            raise ValueError("La medida de profundidad Z debe ser positiva.")

        if dt_medidas is not None:
            self._actualizar_modelo_temporal(dt_medidas)

        # 1. Predicción
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.T + self.Q

        # 2. Corrección
        Z = np.array([[z_val]], dtype=np.float64)
        Y = Z - (self.H @ self.X)
        S = self.H @ self.P @ self.H.T + self.R
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = self.P @ self.H.T @ S_inv
        self.X = self.X + (K @ Y)

        # Actualización por la forma de Joseph y simetrización estricta
        A = self.I - (K @ self.H)
        self.P = A @ self.P @ A.T + (K @ self.R @ K.T)
        self.P = 0.5 * (self.P + self.P.T)

        return self.X[0, 0]


def enviar_velocidad_mavlink(vehiculo, vel_x, vel_y, vel_z):
    if not DRONEKIT_DISPONIBLE or vehiculo is None:
        raise RuntimeError("Conexión con vehículo no disponible o DroneKit inactivo.")
    frame_mavlink = getattr(mavutil.mavlink, MAVLINK_FRAME_VELOCIDAD, None)
    if frame_mavlink is None:
        raise RuntimeError(f"Marco MAVLink no disponible: {MAVLINK_FRAME_VELOCIDAD}")
    msg = vehiculo.message_factory.set_position_target_local_ned_encode(
        0, 0, 0, frame_mavlink,
        MAVLINK_TYPE_MASK_VELOCIDAD_XY, 0, 0, 0, vel_x, vel_y, vel_z, 0, 0, 0, 0, 0)
    vehiculo.send_mavlink(msg)
    vehiculo.flush()


def despachar_orden_mavlink(transmisor, vehiculo, orden_mavlink, limite_max=MAX_VELOCIDAD, fsm=None):
    """
    Pasarela MAVLink robusta y defensiva con contrato inequívoco.
    Retorna: (enviado_ok: bool, estado_transporte: str)
    """
    if orden_mavlink is None:
        return False, "NULO"
    
    try:
        val_orden = validar_escalar_real_finito(orden_mavlink, "orden_mavlink")
        lim_val = validar_escalar_real_finito(limite_max, "limite_max_despacho")
    except ValueError as e:
        return False, f"RECHAZO_VALIDACION ({e})"

    if lim_val <= 0.0:
        return False, "LIMITE_MAX_INVALIDO"

    orden_acotada = float(np.clip(val_orden, -lim_val, lim_val))

    # Barrera en la propia pasarela, no solo en el ciclo de visión: una orden
    # no nula nunca atraviesa el transporte si hay un enclavamiento activo.
    if (fsm is not None and
            (fsm.en_fallo_transporte_latched or fsm.en_emergencia_latched) and
            abs(orden_acotada) > 1e-12):
        return False, "BLOQUEADO_POR_ENCLAVAMIENTO_ACTIVO"
    
    try:
        transmisor(vehiculo, orden_acotada, 0.0, 0.0)
        return True, "OK"
    except Exception as e:
        return False, f"EXCEPCION_TRANSPORTE ({type(e).__name__}: {e})"


# ==========================================
# 6. CALIBRACIÓN ÓPTICA Y VERIFICACIÓN DE RESOLUCIÓN
# ==========================================
def verificar_integridad_calibracion(mtx, dist, resolucion_esperada=RESOLUCION_CALIBRADA,
                                     verificar_correspondencia_runcam=True):
    """Comprueba forma, tipo numérico, finitud y correspondencia con la calibración autorizada."""
    try:
        if mtx is None or dist is None:
            return False
        if not isinstance(mtx, np.ndarray) or not isinstance(dist, np.ndarray):
            return False
        if mtx.shape != (3, 3) or dist.size != 5:
            return False
        if not (np.issubdtype(mtx.dtype, np.number) and np.issubdtype(dist.dtype, np.number)):
            return False
        mtx64 = np.asarray(mtx, dtype=np.float64)
        dist64 = np.asarray(dist, dtype=np.float64)
        if not (np.all(np.isfinite(mtx64)) and np.all(np.isfinite(dist64))):
            return False
        if not np.allclose(mtx64[2], np.array([0.0, 0.0, 1.0]), rtol=0.0, atol=1e-6):
            return False
        fx, fy = float(mtx64[0, 0]), float(mtx64[1, 1])
        cx, cy = float(mtx64[0, 2]), float(mtx64[1, 2])
        ancho, alto = resolucion_esperada
        if fx <= 50.0 or fy <= 50.0:
            return False
        if not (0.2 * ancho <= cx <= 0.8 * ancho and 0.2 * alto <= cy <= 0.8 * alto):
            return False
        if verificar_correspondencia_runcam:
            if not np.allclose(mtx64, MTX_RUNCAM_CALIBRADA, rtol=1e-3, atol=1e-2):
                return False
            if not np.allclose(dist64.reshape(-1), DIST_RUNCAM_CALIBRADA.reshape(-1),
                               rtol=1e-2, atol=1e-3):
                return False
        return True
    except (TypeError, ValueError, IndexError):
        return False


def validar_frame_y_calibracion(frame, cal_matriz_ok, resolucion_esperada=RESOLUCION_CALIBRADA,
                                mtx=None, dist=None):
    """Valida estructura, tipo, canales, resolución y calibración antes de OpenCV."""
    if not isinstance(frame, np.ndarray):
        return False, "FRAME_NO_ES_NUMPY_NDARRAY"
    if frame.ndim != 3:
        return False, "FRAME_DIMENSIONES_INVALIDAS"
    if frame.shape[2] != 3:
        return False, "FRAME_NO_ES_BGR_DE_TRES_CANALES"
    if frame.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32)):
        return False, f"FRAME_DTYPE_NO_COMPATIBLE ({frame.dtype})"
    if frame.size == 0:
        return False, "FRAME_VACIO"
    alto_f, ancho_f = frame.shape[:2]
    if (ancho_f, alto_f) != tuple(resolucion_esperada):
        return False, (
            f"RESOLUCION_INCORRECTA ({ancho_f}x{alto_f} != "
            f"{resolucion_esperada[0]}x{resolucion_esperada[1]})"
        )
    if cal_matriz_ok is not True:
        return False, "BLOQUEO_SIN_NPZ_VERIFICADO"
    if not verificar_integridad_calibracion(
        mtx, dist, resolucion_esperada=resolucion_esperada,
        verificar_correspondencia_runcam=False
    ):
        return False, "BLOQUEO_MATRICES_OPTICAS_NO_VERIFICADAS"
    return True, "OK"


def cargar_calibracion_optica(rutas_busqueda=None, exigir_npz=EXIGIR_NPZ_DESDE_DISCO):
    if rutas_busqueda is None:
        rutas_busqueda = []
        if "__file__" in globals():
            rutas_busqueda.append(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibracion_runcam.npz")
            )
        rutas_busqueda.append("calibracion_runcam.npz")

    for ruta in rutas_busqueda:
        if os.path.exists(ruta):
            try:
                with np.load(ruta) as data:
                    mtx = data["mtx"]
                    dist = data["dist"]
                if verificar_integridad_calibracion(mtx, dist, verificar_correspondencia_runcam=False):
                    return mtx, dist, True, True, f"NPZ_ESTRUCTURA_VERIFICADA_SIN_MODIFICAR ({ruta})"
                else:
                    return (
                        MTX_RUNCAM_CALIBRADA.copy(),
                        DIST_RUNCAM_CALIBRADA.copy(),
                        False,
                        False,
                        f"BLOQUEADO_NPZ_NO_CORRESPONDE ({ruta})"
                    )
            except Exception:
                pass

    mtx_respaldo = MTX_RUNCAM_CALIBRADA.copy()
    dist_respaldo = DIST_RUNCAM_CALIBRADA.copy()
    cal_autorizada = (not exigir_npz) and verificar_integridad_calibracion(mtx_respaldo, dist_respaldo)
    motivo = "RESPALDO_AUTORIZADO" if cal_autorizada else "BLOQUEADO_FALTA_NPZ"
    return mtx_respaldo, dist_respaldo, False, cal_autorizada, motivo


# ==========================================
# 6B. POSE EXTERNA Y VALIDACIÓN EKF3
# ==========================================
def leer_configuracion_geometrica_entorno():
    """Lee geometría explícita; no inventa rumbo de pared ni extrínsecos de cámara."""
    nombres = {
        "rumbo_pared_deg": "AUD05_RUMBO_NORMAL_PARED_DEG",
        "offset_x_m": "AUD05_CAMARA_OFFSET_X_M",
        "offset_y_m": "AUD05_CAMARA_OFFSET_Y_M",
        "offset_z_m": "AUD05_CAMARA_OFFSET_Z_M",
        "origen_ekf_x_wall_m": "AUD05_EKF_ORIGEN_X_WALL_M",
        "origen_ekf_y_wall_m": "AUD05_EKF_ORIGEN_Y_WALL_M",
        "origen_ekf_z_wall_m": "AUD05_EKF_ORIGEN_Z_WALL_M",
    }
    valores = {}
    faltantes = []
    for clave, variable in nombres.items():
        texto = os.environ.get(variable)
        if texto is None or not texto.strip():
            faltantes.append(variable)
            continue
        try:
            valores[clave] = validar_escalar_real_finito(float(texto), variable)
        except (TypeError, ValueError):
            raise ValueError(f"Variable geométrica inválida: {variable}={texto!r}")
    if faltantes:
        raise ValueError("Faltan parámetros geométricos medidos: " + ", ".join(faltantes))
    if not (-360.0 <= valores["rumbo_pared_deg"] <= 360.0):
        raise ValueError("AUD05_RUMBO_NORMAL_PARED_DEG debe estar entre -360 y 360 grados.")
    offset_camara = np.array(
        [valores["offset_x_m"], valores["offset_y_m"], valores["offset_z_m"]], dtype=np.float64
    )
    origen_ekf_wall = np.array(
        [valores["origen_ekf_x_wall_m"], valores["origen_ekf_y_wall_m"], valores["origen_ekf_z_wall_m"]],
        dtype=np.float64
    )
    return valores["rumbo_pared_deg"], offset_camara, origen_ekf_wall


def rotacion_a_euler_zyx(R):
    """Convierte una matriz de rotación válida a roll/pitch/yaw en radianes."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        raise ValueError("Matriz de rotación inválida.")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(R), 1.0, atol=1e-3):
        raise ValueError("La matriz no es una rotación ortonormal válida.")
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    if abs(np.cos(pitch)) > 1e-6:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
        yaw = 0.0
    return roll, pitch, yaw


def calcular_pose_externa_ned(rvec, tvec_cm, rumbo_normal_pared_deg, offset_camara_cuerpo_m, origen_ekf_en_pared_m=None):
    """Calcula pose del origen del cuerpo en NED a partir de solvePnP.

    Suposiciones explícitas: marcador vertical y frontal; marco de pared +X desde pared
    hacia el dron, +Z arriba; rumbo de +X medido horario desde el norte; cuerpo FRD;
    cámara frontal con ejes ópticos (derecha, abajo, delante). El offset debe medirse.
    Los puntos 3D de solvePnP están en centímetros y se convierten aquí a metros.
    origen_ekf_en_pared_m es la posición del origen local EKF expresada en el marco de pared;
    sin ese levantamiento no se puede alinear una pose local con la Pixhawk.
    """
    if rvec is None or tvec_cm is None:
        raise ValueError("Pose solvePnP ausente.")
    rumbo = np.deg2rad(validar_escalar_real_finito(rumbo_normal_pared_deg, "rumbo_normal_pared_deg"))
    offset = np.asarray(offset_camara_cuerpo_m, dtype=np.float64).reshape(-1)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("offset_camara_cuerpo_m debe contener tres valores finitos en metros.")
    if origen_ekf_en_pared_m is None:
        raise ValueError("Falta la posición medida del origen EKF en el marco de pared.")
    origen_ekf_wall = np.asarray(origen_ekf_en_pared_m, dtype=np.float64).reshape(-1)
    if origen_ekf_wall.shape != (3,) or not np.all(np.isfinite(origen_ekf_wall)):
        raise ValueError("origen_ekf_en_pared_m debe contener tres valores finitos en metros.")
    R_camera_marker, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t_camera_marker = np.asarray(tvec_cm, dtype=np.float64).reshape(3) / 100.0
    if not np.all(np.isfinite(t_camera_marker)) or t_camera_marker[2] <= 0.0:
        raise ValueError("La pose del marcador está detrás de la cámara o no es finita.")
    # El marcador se ha definido vertical y mirando a la cámara: su normal +Z local
    # debe apuntar aproximadamente hacia -Z óptico. Rechazar poses volteadas/oblicuas.
    if float(R_camera_marker[2, 2]) > -0.5:
        raise ValueError("Orientación ArUco incompatible con marcador frontal vertical.")

    T_camera_marker = np.eye(4, dtype=np.float64)
    T_camera_marker[:3, :3] = R_camera_marker
    T_camera_marker[:3, 3] = t_camera_marker

    # Ejes del marcador: +x horizontal, +y hacia arriba, +z saliendo hacia la cámara.
    # En el marco de pared: +x_wall hacia el dron, +y_wall paralelo a pared, +z_wall arriba.
    R_wall_marker = np.array([[0.0, 0.0, 1.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]], dtype=np.float64)
    T_wall_marker = np.eye(4, dtype=np.float64)
    T_wall_marker[:3, :3] = R_wall_marker
    T_wall_marker[:3, 3] = np.array([0.0, 0.0, ALTURA_CENTRO_ARUCO_M], dtype=np.float64)
    T_wall_camera = T_wall_marker @ np.linalg.inv(T_camera_marker)

    # Convierte marco local de pared (X hacia fuera, Y paralelo a pared, Z arriba)
    # a NED usando el rumbo medido del eje +X_wall.
    c, si = float(np.cos(rumbo)), float(np.sin(rumbo))
    R_ned_wall = np.array([[c, si, 0.0],
                           [si, -c, 0.0],
                           [0.0, 0.0, -1.0]], dtype=np.float64)
    T_ned_wall = np.eye(4, dtype=np.float64)
    T_ned_wall[:3, :3] = R_ned_wall
    # La posición se expresa respecto al origen EKF medido, no se asume que coincida
    # automáticamente con la proyección del marcador sobre el suelo.
    T_ned_wall[:3, 3] = -R_ned_wall @ origen_ekf_wall
    T_ned_camera = T_ned_wall @ T_wall_camera

    # R_body_camera: ejes de cámara expresados en cuerpo FRD; cámara alineada al frente.
    R_body_camera = np.array([[0.0, 0.0, 1.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]], dtype=np.float64)
    T_body_camera = np.eye(4, dtype=np.float64)
    T_body_camera[:3, :3] = R_body_camera
    T_body_camera[:3, 3] = offset  # posición de la lente expresada en el cuerpo, metros
    T_ned_body = T_ned_camera @ np.linalg.inv(T_body_camera)

    roll, pitch, yaw = rotacion_a_euler_zyx(T_ned_body[:3, :3])
    pos = T_ned_body[:3, 3]
    if not np.all(np.isfinite(pos)):
        raise ValueError("La pose transformada contiene valores no finitos.")
    return {
        "x_ned_m": float(pos[0]), "y_ned_m": float(pos[1]), "z_ned_m": float(pos[2]),
        "roll_rad": roll, "pitch_rad": pitch, "yaw_rad": yaw,
        "frame": "LOCAL_NED", "units": "m_rad",
    }


def verificar_configuracion_ekf_externa(vehiculo, exigir_confirmacion=True):
    """Chequea parámetros y estado EKF; la evidencia de fusión debe confirmarse en Mission Planner/logs."""
    if vehiculo is None:
        return False, "VEHICULO_AUSENTE"
    if exigir_confirmacion and os.environ.get("AUD05_EKF_FUSION_CONFIRMADA") != "1":
        return False, "FALTA_CONFIRMACION_BANCO_EKF_FUSION"
    try:
        parametros = vehiculo.parameters
        esperados = {
            "VISO_TYPE": (3.0,),
            "VISO_POS_X": (0.0,), "VISO_POS_Y": (0.0,), "VISO_POS_Z": (0.0,),
            "EK3_SRC1_POSXY": (6.0,),
            "EK3_SRC1_VELXY": (6.0,),
            "EK3_SRC1_POSZ": (1.0, 6.0),
            "EK3_SRC1_VELZ": (6.0,),
            "EK3_SRC1_YAW": (1.0, 6.0),
        }
        for nombre, permitidos in esperados.items():
            try:
                valor = float(parametros[nombre])
            except Exception:
                return False, f"PARAMETRO_EKF_NO_DISPONIBLE:{nombre}"
            if not np.isfinite(valor) or not any(abs(valor - permitido) < 1e-6 for permitido in permitidos):
                return False, f"PARAMETRO_EKF_INCOMPATIBLE:{nombre}={valor}"
        if getattr(vehiculo, "last_heartbeat", None) is None or float(vehiculo.last_heartbeat) >= 2.0:
            return False, "HEARTBEAT_ANTIGUO"
        if not bool(getattr(vehiculo, "ekf_ok", False)):
            return False, "EKF_NO_REPORTA_ESTADO_OK"
        local = getattr(getattr(vehiculo, "location", None), "local_frame", None)
        if local is None or not all(np.isfinite(float(getattr(local, eje, np.nan))) for eje in ("north", "east", "down")):
            return False, "POSICION_LOCAL_NO_DISPONIBLE"
        return True, "EKF_CONFIGURADO_Y_ESTADO_OK; FUSION_REQUIERE_EVIDENCIA_MISSION_PLANNER"
    except Exception as exc:
        return False, f"ERROR_COMPROBACION_EKF:{type(exc).__name__}:{exc}"


class PublicadorVisionMavlink:
    """Publica VISION_POSITION_ESTIMATE y velocidad derivada, separado del control de movimiento."""
    def __init__(self, frecuencia_max_hz=FRECUENCIA_VISION_OBJETIVO_HZ):
        self.frecuencia_max_hz = validar_escalar_real_finito(frecuencia_max_hz, "frecuencia_max_hz")
        if self.frecuencia_max_hz < FRECUENCIA_VISION_MIN_HZ:
            raise ValueError("La frecuencia máxima de visión debe ser >= 4 Hz.")
        self.ultima_pose = None
        self.ultimo_t = None
        self.ultimo_envio = None
        self.poses_publicadas = 0

    def publicar(self, vehiculo, pose, t_monotonic=None):
        if not DRONEKIT_DISPONIBLE or vehiculo is None:
            raise RuntimeError("No hay enlace MAVLink disponible para publicar visión externa.")
        requerido = ("x_ned_m", "y_ned_m", "z_ned_m", "roll_rad", "pitch_rad", "yaw_rad")
        valores = [validar_escalar_real_finito(pose[k], k) for k in requerido]
        ahora = time.monotonic() if t_monotonic is None else validar_escalar_real_finito(t_monotonic, "t_monotonic")
        if self.ultimo_envio is not None:
            dt_envio = ahora - self.ultimo_envio
            if dt_envio > TIMEOUT_POSE_EXTERNA_S:
                raise RuntimeError(f"FRECUENCIA_VISION_INSUFICIENTE ({dt_envio:.3f}s > {TIMEOUT_POSE_EXTERNA_S:.3f}s)")
            if dt_envio < 1.0 / self.frecuencia_max_hz:
                return False, "LIMITADO_FRECUENCIA"
        usec = int(time.time_ns() // 1000)  # timestamp absoluto permitido por MAVLink; mantener sincronía del PC.
        x, y, z, roll, pitch, yaw = valores
        cov_pos = [float("nan")] * 21  # incertidumbre no caracterizada: no fingir covarianza nula.
        try:
            msg_pos = mavutil.mavlink.MAVLink_vision_position_estimate_message(
                usec, x, y, z, roll, pitch, yaw, cov_pos, 0
            )
        except TypeError:
            msg_pos = mavutil.mavlink.MAVLink_vision_position_estimate_message(
                usec, x, y, z, roll, pitch, yaw
            )
        vehiculo.send_mavlink(msg_pos)

        # Derivar velocidad solo de dos poses reales recientes; nunca generar velocidad a partir de pose ausente.
        if self.ultima_pose is not None and self.ultimo_t is not None:
            dt = ahora - self.ultimo_t
            if 0.0 < dt <= TIMEOUT_POSE_EXTERNA_S:
                vel = [(pose[k] - self.ultima_pose[k]) / dt for k in ("x_ned_m", "y_ned_m", "z_ned_m")]
                if np.all(np.isfinite(vel)):
                    cov_vel = [float("nan")] * 9
                    try:
                        msg_vel = mavutil.mavlink.MAVLink_vision_speed_estimate_message(
                            usec, float(vel[0]), float(vel[1]), float(vel[2]), cov_vel
                        )
                    except TypeError:
                        msg_vel = mavutil.mavlink.MAVLink_vision_speed_estimate_message(
                            usec, float(vel[0]), float(vel[1]), float(vel[2])
                        )
                    vehiculo.send_mavlink(msg_vel)
        vehiculo.flush()
        self.ultima_pose = {k: pose[k] for k in ("x_ned_m", "y_ned_m", "z_ned_m")}
        self.ultimo_t = ahora
        self.ultimo_envio = ahora
        self.poses_publicadas += 1
        return True, "POSE_Y_VELOCIDAD_PUBLICADAS"

    def invalidar_historial(self):
        self.ultima_pose = None
        self.ultimo_t = None
        self.ultimo_envio = None
        self.poses_publicadas = 0


# ==========================================
# 7. PASO LÓGICO INTEGRADO Y CICLO DE CONTROL
# ==========================================
def hay_transicion_guided(modo_anterior, modo_pixhawk):
    return (modo_anterior == "GUIDED") != (modo_pixhawk == "GUIDED")


def ejecutar_paso_logico(fsm, kalman, distancia_anterior, modo_anterior, modo_pixhawk,
                         aruco_valido_en_frame, z_cruda_medida, t_actual, calibracion_valida=True,
                         dt_medidas=0.033, navegacion_ekf_valida=True):
    estado_previo = fsm.estado

    if modo_anterior == "GUIDED" and modo_pixhawk != "GUIDED" and fsm.estado == EstadoVuelo.CONTROL_ACTIVO:
        # Cambio de modo durante control activo: se pierde la autoridad esperada; enclavar seguridad.
        fsm.forzar_emergencia(transporte_roto=False)
        aruco_valido_en_frame = False
        z_cruda_medida = None
    elif hay_transicion_guided(modo_anterior, modo_pixhawk):
        fsm.reiniciar_a_busqueda()
        distancia_anterior = None
        if modo_pixhawk != "GUIDED":
            aruco_valido_en_frame = False
            z_cruda_medida = None

    if not calibracion_valida and fsm.estado == EstadoVuelo.CONTROL_ACTIVO:
        fsm.forzar_emergencia(transporte_roto=False)
        aruco_valido_en_frame = False

    estado_actual = fsm.actualizar(aruco_valido_en_frame, t_actual, navegacion_ekf_valida=navegacion_ekf_valida)
    velocidad_segura = 0.0

    if estado_actual == EstadoVuelo.BUSQUEDA:
        distancia_anterior = None
        distancia_filtrada = kalman.X[0, 0]

    elif estado_actual == EstadoVuelo.VALIDACION:
        if aruco_valido_en_frame and z_cruda_medida is not None:
            if fsm.frames_consecutivos == 1:
                distancia_filtrada = kalman.reiniciar(z_cruda_medida)
            else:
                distancia_filtrada = kalman.actualizar(z_cruda_medida, dt_medidas=dt_medidas)
            distancia_anterior = z_cruda_medida
        else:
            distancia_filtrada = kalman.X[0, 0]

    elif estado_actual == EstadoVuelo.CONTROL_ACTIVO:
        if aruco_valido_en_frame and z_cruda_medida is not None:
            distancia_anterior = z_cruda_medida
            distancia_filtrada = kalman.actualizar(z_cruda_medida, dt_medidas=dt_medidas)
            factor_visibilidad = 1.0
        else:
            distancia_filtrada = kalman.X[0, 0]
            factor_visibilidad = max(0.0, 1.0 - (fsm.tiempo_ciego / fsm.timeout_s))

        error_z = distancia_filtrada - DISTANCIA_OBJETIVO_CM
        velocidad_comandada = Kp_z * error_z * factor_visibilidad
        velocidad_segura = max(min(velocidad_comandada, MAX_VELOCIDAD), -MAX_VELOCIDAD)

    elif estado_actual == EstadoVuelo.EMERGENCIA:
        distancia_anterior = None
        distancia_filtrada = kalman.X[0, 0]
        velocidad_segura = 0.0

    control_autonomo_autorizado = (
        (modo_pixhawk == "GUIDED")
        and fsm.referencia_visual_validada
        and bool(calibracion_valida)
    )
    if modo_pixhawk == "GUIDED":
        if control_autonomo_autorizado:
            orden_mavlink = velocidad_segura
        elif estado_actual == EstadoVuelo.EMERGENCIA:
            # Intento de cero en estado de seguridad; no significa parada física confirmada.
            orden_mavlink = 0.0
        else:
            # En BUSQUEDA/VALIDACION no se envían consignas de movimiento.
            orden_mavlink = None
    else:
        orden_mavlink = None

    transicion_str = f"{estado_previo.name}->{estado_actual.name}" if estado_actual != estado_previo else "-"
    return estado_actual, distancia_anterior, distancia_filtrada, velocidad_segura, orden_mavlink, transicion_str


def ejecutar_ciclo_control_integrado(
    fsm,
    kalman,
    validador,
    mtx,
    dist,
    puntos_3d,
    frame,
    modo_pixhawk,
    modo_anterior,
    t_actual,
    calibracion_autorizada_ok,
    distancia_anterior_in,
    dt_medidas_in,
    transmisor_mavlink,
    vehiculo_ref,
    detector=None,
    id_objetivo=ID_ARUCO_OBJETIVO,
    rumbo_normal_pared_deg=None,
    offset_camara_cuerpo_m=None,
    origen_ekf_en_pared_m=None,
    publicador_vision=None,
    navegacion_ekf_valida=True,
):
    """Procesa una iteración y registra por separado orden solicitada, parada y resultado de transporte."""
    estado_previo_ciclo = fsm.estado
    motivo_rechazo = "OK"

    dt_valido = True
    try:
        dt_ciclo = validar_escalar_real_finito(dt_medidas_in, "dt_medidas_in")
        if dt_ciclo <= 0.0:
            raise ValueError("dt_medidas_in debe ser estrictamente positivo.")
    except (ValueError, TypeError) as exc:
        dt_valido = False
        dt_ciclo = None
        motivo_rechazo = f"DT_INVALIDO ({type(exc).__name__}: {exc})"

    if not dt_valido:
        # Fail-closed: no conversión, detección, filtrado ni control normal.
        fsm.forzar_emergencia(transporte_roto=False)
        calibracion_valida = False
    elif dt_ciclo >= MAX_DT_CICLO_S:
        fsm.forzar_emergencia(transporte_roto=False)
        motivo_rechazo = (
            f"PAUSA_CICLO_EXCESIVA ({dt_ciclo:.3f}s >= {MAX_DT_CICLO_S:.3f}s)"
        )
        calibracion_valida = False
    else:
        calibracion_valida, motivo_cal_frame = validar_frame_y_calibracion(
            frame, calibracion_autorizada_ok, mtx=mtx, dist=dist
        )
        if not calibracion_valida:
            motivo_rechazo = f"RECHAZO_CALIBRACION_OR_FRAME ({motivo_cal_frame})"

    gray = None
    if frame is not None and calibracion_valida and not fsm.en_emergencia_latched:
        try:
            if not hasattr(frame, "shape") or len(frame.shape) < 2:
                motivo_rechazo = "FRAME_INVALIDO_SHAPE"
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            motivo_rechazo = f"EXCEPCION_CV_CONVERT ({type(e).__name__}: {e})"

    aruco_valido = False
    z_cruda = None
    pose_externa_ned = None
    rvec_valido = None
    tvec_valido = None

    if gray is not None and calibracion_valida and motivo_rechazo == "OK":
        try:
            if detector is None:
                detector = crear_detector_aruco()

            esq, ids, _ = detector.detectMarkers(gray)
            esq_2d, _, motivo_sel = seleccionar_marcador_objetivo(esq, ids, id_objetivo)

            if motivo_sel != "OK":
                motivo_rechazo = motivo_sel
            elif esq_2d is not None:
                if (
                    puntos_3d is None or not isinstance(puntos_3d, np.ndarray) or puntos_3d.shape != (4, 3)
                    or esq_2d is None or not isinstance(esq_2d, np.ndarray) or esq_2d.size != 8
                    or not verificar_integridad_calibracion(mtx, dist, verificar_correspondencia_runcam=False)
                ):
                    motivo_rechazo = "RECHAZO_SOLVE_PNP_DIMENSIONES_O_CALIBRACION_INVALIDAS"
                else:
                    esq_solve = np.asarray(esq_2d, dtype=np.float32).reshape(4, 2)
                    exito, rvec, tvec = cv2.solvePnP(
                        np.asarray(puntos_3d, dtype=np.float32), esq_solve,
                        np.asarray(mtx, dtype=np.float64), np.asarray(dist, dtype=np.float64)
                    )
                    if not exito or rvec is None or tvec is None:
                        motivo_rechazo = "RECHAZO_SOLVE_PNP_FAIL"
                    else:
                        z_candidata = float(tvec.flatten()[2])
                        ok_aud, motivo_aud = validador.auditar(
                            z_candidata, distancia_anterior_in, puntos_3d, esq_solve,
                            rvec, tvec, mtx, dist, fsm.estado
                        )
                        if ok_aud:
                            try:
                                if (rumbo_normal_pared_deg is not None and offset_camara_cuerpo_m is not None
                                        and origen_ekf_en_pared_m is not None):
                                    pose_externa_ned = calcular_pose_externa_ned(
                                        rvec, tvec, rumbo_normal_pared_deg, offset_camara_cuerpo_m,
                                        origen_ekf_en_pared_m
                                    )
                                aruco_valido = True
                                z_cruda = z_candidata
                                rvec_valido = rvec
                                tvec_valido = tvec
                                motivo_rechazo = motivo_aud
                            except Exception as exc_pose:
                                motivo_rechazo = f"RECHAZO_POSE_EXTERNA ({type(exc_pose).__name__}: {exc_pose})"
                        else:
                            motivo_rechazo = motivo_aud
        except Exception as e:
            motivo_rechazo = f"EXCEPCION_VISION_ARUCO ({type(e).__name__}: {e})"

    estado_actual, dist_ant_out, dist_filt, vel_seg, orden_mav_solicitada, trans_str = ejecutar_paso_logico(
        fsm=fsm, kalman=kalman, distancia_anterior=distancia_anterior_in,
        modo_anterior=modo_anterior, modo_pixhawk=modo_pixhawk,
        aruco_valido_en_frame=aruco_valido, z_cruda_medida=z_cruda,
        t_actual=t_actual, calibracion_valida=calibracion_valida,
        dt_medidas=(min(max(dt_ciclo, 0.001), MAX_DT_CICLO_S - EPSILON_TIEMPO_S)
                    if dt_valido else 0.033),
        navegacion_ekf_valida=navegacion_ekf_valida
    )

    # Publicar visión externa solo tras 10 observaciones consecutivas válidas.
    # Es un canal separado del control de velocidad y no se publican poses inventadas.
    estado_vision_tx = "NO_PUBLICADA"
    if (publicador_vision is not None and aruco_valido and pose_externa_ned is not None
            and fsm.frames_consecutivos >= FRAMES_VALIDACION_REQ
            and fsm.estado in (EstadoVuelo.VALIDACION, EstadoVuelo.CONTROL_ACTIVO)):
        try:
            ok_vision, estado_vision_tx = publicador_vision.publicar(vehiculo_ref, pose_externa_ned, t_monotonic=t_actual)
            if not ok_vision and estado_vision_tx != "LIMITADO_FRECUENCIA":
                raise RuntimeError(estado_vision_tx)
        except Exception as exc_vision:
            fsm.forzar_emergencia(transporte_roto=True)
            estado_actual = EstadoVuelo.EMERGENCIA
            vel_seg = 0.0
            orden_mav_solicitada = 0.0 if modo_pixhawk == "GUIDED" else None
            motivo_rechazo = f"FALLO_PUBLICACION_VISION ({type(exc_vision).__name__}: {exc_vision})"
            trans_str = f"{estado_previo_ciclo.name}->{EstadoVuelo.EMERGENCIA.name}"
            if publicador_vision is not None:
                publicador_vision.invalidar_historial()

    fallo_transporte_previo = fsm.en_fallo_transporte_latched
    if fallo_transporte_previo and orden_mav_solicitada is not None:
        # Barrera independiente de la FSM: no emitir movimiento tras fallo enclavado.
        ok_envio, estado_trans = False, "BLOQUEADO_POR_ENCLAVAMIENTO_PREVIO"
    else:
        ok_envio, estado_trans = despachar_orden_mavlink(
            transmisor_mavlink, vehiculo_ref, orden_mav_solicitada, fsm=fsm
        )
    ok_compensacion = None
    estado_compensacion = "NO_REQUERIDA"
    orden_mav_efectiva = orden_mav_solicitada

    if orden_mav_solicitada is not None and (not ok_envio or fallo_transporte_previo):
        if not ok_envio and not fallo_transporte_previo:
            fsm.forzar_emergencia(transporte_roto=True)
            if modo_pixhawk == "GUIDED":
                ok_compensacion, estado_compensacion = despachar_orden_mavlink(
                    transmisor_mavlink, vehiculo_ref, 0.0, fsm=fsm
                )
            else:
                estado_compensacion = "NO_GUIDED"
        elif fallo_transporte_previo:
            estado_compensacion = "BLOQUEADO_POR_ENCLAVAMIENTO_PREVIO"

        estado_actual = fsm.estado
        vel_seg = 0.0
        orden_mav_efectiva = 0.0 if modo_pixhawk == "GUIDED" else None
        dist_ant_out = None
        trans_str = (
            f"{estado_previo_ciclo.name}->{estado_actual.name}"
            if estado_actual != estado_previo_ciclo else "-"
        )

    if not ok_envio and orden_mav_solicitada is not None and not fallo_transporte_previo:
        resultado_transporte = "ERROR_TRANSPORTE_ACTUAL"
    elif fallo_transporte_previo or fsm.en_fallo_transporte_latched:
        resultado_transporte = "BLOQUEADO_POR_FALLO_PREVIO"
    elif ok_envio:
        resultado_transporte = "OK_ENVIADO"
    else:
        resultado_transporte = "NO_ENVIADO"

    if fsm.estado == EstadoVuelo.EMERGENCIA:
        estado_actual = EstadoVuelo.EMERGENCIA
        vel_seg = 0.0
        if modo_pixhawk == "GUIDED":
            orden_mav_efectiva = 0.0
        dist_ant_out = None
        trans_str = (
            f"{estado_previo_ciclo.name}->{EstadoVuelo.EMERGENCIA.name}"
            if estado_previo_ciclo != EstadoVuelo.EMERGENCIA else "-"
        )

    return {
        "estado_final": estado_actual,
        "distancia_anterior": dist_ant_out,
        "distancia_filtrada": dist_filt,
        "velocidad_segura": vel_seg,
        "orden_mavlink": orden_mav_efectiva,
        "orden_mavlink_solicitada": orden_mav_solicitada,
        "ok_envio": ok_envio,
        "estado_transporte": estado_trans,
        "resultado_transporte": resultado_transporte,
        "ok_compensacion": ok_compensacion,
        "estado_compensacion": estado_compensacion,
        "z_candidata": z_cruda,
        "motivo_rechazo": motivo_rechazo,
        "pose_externa_ned": pose_externa_ned,
        "estado_vision_tx": estado_vision_tx,
        "transicion": trans_str,
    }


# ==========================================
# 8. BATERÍA DE PRUEBAS AUTOMATIZADAS
# ==========================================
def ejecutar_pruebas_unitarias():
    print("=" * 80)
    print("EJECUTANDO BATERÍA DE PRUEBAS AUTOMATIZADAS - AUD-05 v5.22")
    print("=" * 80)

    mtx_t = MTX_RUNCAM_CALIBRADA.astype(np.float32)
    dist_t = DIST_RUNCAM_CALIBRADA.astype(np.float32)
    pts_3d = np.array([
        [-LADO_REAL_CM / 2,  LADO_REAL_CM / 2, 0],
        [ LADO_REAL_CM / 2,  LADO_REAL_CM / 2, 0],
        [ LADO_REAL_CM / 2, -LADO_REAL_CM / 2, 0],
        [-LADO_REAL_CM / 2, -LADO_REAL_CM / 2, 0]
    ], dtype=np.float32)

    def nuevo_contexto():
        return (
            FSM_Seguridad(),
            ValidadorVisual(),
            FiltroKalmanCinematico(0.033, 0.1, 0.05, 8.0, 120.0)
        )

    def paso_sim(fsm, val, kal, dist_ant, modo_ant, modo_pix, ret_cam, z_sim, ruido_px, t_now, cal_ok=True, dt_medidas=0.033):
        if not ret_cam:
            est = fsm.forzar_emergencia(transporte_roto=False)
            ord_m = 0.0 if modo_pix == "GUIDED" else None
            return est, None, kal.X[0, 0], 0.0, ord_m, "FALLO_CAMARA", modo_pix
        valido = False
        z_valida = None
        motivo = "ArUco NO detectado"
        estado_efectivo = (
            EstadoVuelo.BUSQUEDA
            if hay_transicion_guided(modo_ant, modo_pix)
            else fsm.estado
        )
        if z_sim is not None and cal_ok:
            rvec = np.zeros((3, 1), dtype=np.float32)
            tvec = np.array([[0.0], [0.0], [float(z_sim)]], dtype=np.float32)
            pts, _ = cv2.projectPoints(pts_3d, rvec, tvec, mtx_t, dist_t)
            esq = pts.reshape(-1, 2).astype(np.float32)
            if ruido_px > 0.0:
                esq[:, 0] += ruido_px
            ref = dist_ant if estado_efectivo in [EstadoVuelo.VALIDACION, EstadoVuelo.CONTROL_ACTIVO] else None
            ok, motivo = val.auditar(float(z_sim), ref, pts_3d, esq, rvec, tvec, mtx_t, dist_t, estado_efectivo)
            if ok:
                valido = True
                z_valida = float(z_sim)
        est, dist_ant, z_f, v_seg, ord_m, _ = ejecutar_paso_logico(
            fsm, kal, dist_ant, modo_ant, modo_pix, valido, z_valida, t_now, calibracion_valida=cal_ok, dt_medidas=dt_medidas
        )
        return est, dist_ant, z_f, v_seg, ord_m, motivo, modo_pix

    def ut_01():
        fsm, val, kal = nuevo_contexto()
        lanzo_error = False
        try:
            kal.actualizar(None)
        except (ValueError, TypeError):
            lanzo_error = True
        assert lanzo_error, "UT-01a"
        dist_ant, modo_ant, t = None, "GUIDED", 0.0
        for _ in range(10):
            t += 0.033
            est, dist_ant, z_f, _, _, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 100.0, 0.0, t, dt_medidas=0.033)
        z_ref = z_f
        t += 0.033
        est, dist_ant, z_ciego, _, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, None, 0.0, t, dt_medidas=0.033)
        assert est == EstadoVuelo.EMERGENCIA and z_ciego == z_ref, "UT-01b"
        assert ord_m == 0.0
        return "UT-01 [Aislada]: None no actualiza Kalman y pérdida en control enclava seguridad"

    def ut_02():
        fsm, val, kal = nuevo_contexto()
        dist_ant, modo_ant, t = None, "GUIDED", 0.0
        for _ in range(10):
            t += 0.033
            est, dist_ant, _, _, _, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 100.0, 0.0, t, dt_medidas=0.033)
        t += 0.033
        est, dist_ant, _, _, ord_m, mot, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, None, 0.0, t, dt_medidas=0.033)
        assert est == EstadoVuelo.EMERGENCIA and ord_m == 0.0, "UT-02"
        t += 0.033
        est, dist_ant, _, _, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 100.0, 0.0, t, dt_medidas=0.033)
        assert est == EstadoVuelo.EMERGENCIA and ord_m == 0.0, "UT-02b"
        return "UT-02 [Aislada]: La visión no recupera automáticamente un estado de seguridad enclavado"

    def ut_03():
        fsm, val, kal = nuevo_contexto()
        est, dist_ant, _, _, _, _, modo_ant = paso_sim(fsm, val, kal, None, "GUIDED", "GUIDED", True, 90.0, 0.0, 1.0, dt_medidas=0.033)
        assert est == EstadoVuelo.VALIDACION and fsm.frames_consecutivos == 1, "UT-03a"
        est, dist_ant, _, _, _, mot, _ = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 180.0, 0.0, 1.033, dt_medidas=0.033)
        assert est == EstadoVuelo.BUSQUEDA and "Rechazo Cinemático" in mot, "UT-03b"
        return "UT-03 [Aislada]: Rechazo de salto cinemático (> 40 cm) durante VALIDACION"

    def ut_04():
        fsm, _, _ = nuevo_contexto()
        fsm.forzar_emergencia()
        rearme_ok = fsm.rearmar_sistema_supervisado(autorizacion_operador=True, enlace_verificado=True, vehiculo_desarmado=True, modo_verificado=True, calibracion_autorizada=True, transporte_verificado=True)
        assert rearme_ok is True, "UT-04-rearme: rearme válido rechazado"
        est = fsm.actualizar(True, 5.0)
        assert est == EstadoVuelo.VALIDACION, "UT-04a"
        est = fsm.actualizar(False, 5.033)
        assert est == EstadoVuelo.BUSQUEDA, "UT-04b"
        return "UT-04 [Aislada]: Memoria de emergencia y transición a BUSQUEDA ante frame inválido post-rearme"

    def ut_05():
        fsm, val, kal = nuevo_contexto()
        kal.X[1, 0] = 30.0
        est, _, _, _, _, _, _ = paso_sim(fsm, val, kal, None, "GUIDED", "GUIDED", True, 95.0, 0.0, 1.0, dt_medidas=0.033)
        assert est == EstadoVuelo.VALIDACION and kal.X[1, 0] == 0.0 and np.allclose(kal.P, np.eye(2)), "UT-05"
        return "UT-05 [Aislada]: Limpieza de velocidad residual X[1,0]=0.0 y P=I en frame 1/10"

    def ut_06():
        fsm, val, kal = nuevo_contexto()
        dist_ant, modo_ant, t = None, "ALT_HOLD", 0.0
        for _ in range(100):
            t += 0.033
            est, dist_ant, _, _, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "ALT_HOLD", True, None, 0.0, t, dt_medidas=0.033)
            assert est == EstadoVuelo.BUSQUEDA and ord_m is None, "UT-06"
        return "UT-06 [Aislada]: 100 fotogramas sin ArUco en BUSQUEDA sin entrar en EMERGENCIA"

    def ut_07():
        fsm, val, kal = nuevo_contexto()
        dist_ant, modo_ant, t = None, "GUIDED", 0.0
        for _ in range(9):
            t += 0.033
            est, dist_ant, _, _, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 100.0, 0.0, t, dt_medidas=0.033)
        assert est == EstadoVuelo.VALIDACION and fsm.frames_consecutivos == 9 and ord_m is None, "UT-07a"
        t += 0.033
        est, dist_ant, _, _, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, None, 0.0, t, dt_medidas=0.033)
        assert est == EstadoVuelo.BUSQUEDA and dist_ant is None and ord_m is None, "UT-07b"
        for _ in range(9):
            t += 0.033
            est, dist_ant, _, _, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 100.0, 0.0, t, dt_medidas=0.033)
        assert est == EstadoVuelo.VALIDACION and fsm.frames_consecutivos == 9 and ord_m is None, "UT-07c"
        t += 0.400
        est, dist_ant, _, _, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 100.0, 0.0, t, dt_medidas=0.400)
        assert est == EstadoVuelo.VALIDACION and fsm.frames_consecutivos == 1 and ord_m is None, "UT-07d"
        return "UT-07 [Aislada]: Secuencia 9+1+9 y reinicio si dt > 0.25 s durante VALIDACION"

    def ut_08():
        fsm, val, kal = nuevo_contexto()
        dist_ant, modo_ant, t = None, "ALT_HOLD", 0.0
        for _ in range(10):
            t += 0.033
            est, dist_ant, _, v_s, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "ALT_HOLD", True, 100.0, 0.0, t, dt_medidas=0.033)
        assert est == EstadoVuelo.CONTROL_ACTIVO and v_s > 0.0 and ord_m is None, "UT-08"
        return "UT-08 [Aislada]: Validación en ALT_HOLD separa velocidad calculada de orden MAVLink (None)"

    def ut_09():
        fsm, val, kal = nuevo_contexto()
        dist_ant, modo_ant, t = None, "GUIDED", 0.0
        for _ in range(10):
            t += 0.033
            est, dist_ant, z_f, _, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 100.0, 0.0, t, dt_medidas=0.033)
        z_antes = z_f
        v_prev = ord_m
        t += 0.033
        est, dist_ant, z_f, _, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, None, 0.0, t, dt_medidas=0.033)
        assert est == EstadoVuelo.EMERGENCIA and z_f == z_antes and ord_m == 0.0, "UT-09"
        return "UT-09 [Aislada]: Pérdida visual durante control no reutiliza pose antigua para ordenar movimiento"

    def ut_10():
        fsm, val, kal = nuevo_contexto()
        dist_ant, modo_ant, t = None, "GUIDED", 1.0
        for _ in range(10):
            t += 0.033
            est, dist_ant, _, _, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 100.0, 0.0, t, dt_medidas=0.033)
        t_ref = t
        est, dist_ant, _, _, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, None, 0.0, t_ref + 0.490, dt_medidas=0.490)
        assert est == EstadoVuelo.EMERGENCIA and ord_m == 0.0 and dist_ant is None, "UT-10a"
        return "UT-10 [Aislada]: Transición inmediata a EMERGENCIA en la primera pérdida de visión en CONTROL_ACTIVO"

    def ut_11():
        fsm, val, kal = nuevo_contexto()
        dist_ant, modo_ant, t = None, "GUIDED", 0.0
        for _ in range(10):
            t += 0.033
            est, dist_ant, _, _, _, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 100.0, 0.0, t, dt_medidas=0.033)
        t += 0.550
        est, dist_ant, _, _, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, None, 0.0, t, dt_medidas=0.550)
        assert est == EstadoVuelo.EMERGENCIA and ord_m == 0.0, "UT-11 previa"
        fsm.rearmar_sistema_supervisado(autorizacion_operador=True, enlace_verificado=True, vehiculo_desarmado=True, modo_verificado=True, calibracion_autorizada=True, transporte_verificado=True)
        t += 0.033
        est, dist_ant, _, _, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 95.0, 0.0, t, dt_medidas=0.033)
        assert est == EstadoVuelo.VALIDACION and fsm.frames_consecutivos == 1 and ord_m is None, "UT-11a"
        t += 0.033
        est, dist_ant, _, _, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, None, 0.0, t, dt_medidas=0.033)
        assert est == EstadoVuelo.BUSQUEDA and ord_m is None, "UT-11b: Retorno a BUSQUEDA ante frame inválido post-rearme"
        return "UT-11 [Aislada]: En EMERGENCIA, rearme y 1 válido + 1 inválido regresa a BUSQUEDA"

    def ut_12():
        fsm, val, kal = nuevo_contexto()
        dist_ant, modo_ant, t = None, "GUIDED", 0.0
        for _ in range(10):
            t += 0.033
            est, dist_ant, _, _, _, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 100.0, 0.0, t, dt_medidas=0.033)
        t += 0.550
        est, dist_ant, _, _, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, None, 0.0, t, dt_medidas=0.550)
        assert est == EstadoVuelo.EMERGENCIA, "UT-12 previa"
        fsm.rearmar_sistema_supervisado(autorizacion_operador=True, enlace_verificado=True, vehiculo_desarmado=True, modo_verificado=True, calibracion_autorizada=True, transporte_verificado=True)
        for i in range(1, 11):
            t += 0.033
            est, dist_ant, _, _, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 95.0, 0.0, t, dt_medidas=0.033)
            if i < 10:
                assert est == EstadoVuelo.VALIDACION and ord_m is None, f"UT-12 frame {i}"
            else:
                assert est == EstadoVuelo.CONTROL_ACTIVO and ord_m > 0.0, "UT-12 frame 10"
        return "UT-12 [Aislada]: Revalidación de 10 fotogramas consecutivos tras rearme supervisado"

    def ut_13():
        fsm, val, kal = nuevo_contexto()
        dist_ant, modo_ant, t = None, "GUIDED", 0.0
        for _ in range(10):
            t += 0.033
            est, dist_ant, z_f, _, _, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 95.0, 0.0, t, dt_medidas=0.033)
        z_ref = z_f
        t += 0.033
        est, dist_ant, z_f, _, ord_m, mot, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 220.0, 0.0, t, dt_medidas=0.033)
        assert "Rechazo Cinemático" in mot and z_f == z_ref and dist_ant is None, "UT-13a"
        assert est == EstadoVuelo.EMERGENCIA and ord_m == 0.0, "UT-13b"
        # Un outlier de reproyección en VALIDACION no debe actualizar el Kalman.
        fsm2, val2, kal2 = nuevo_contexto()
        dist2, modo2, t2 = None, "GUIDED", 0.0
        t2 += 0.033
        _, dist2, z2, _, _, _, modo2 = paso_sim(fsm2, val2, kal2, dist2, modo2, "GUIDED", True, 95.0, 0.0, t2, dt_medidas=0.033)
        z2_ref = z2
        t2 += 0.033
        est2, dist2, z2, _, _, mot2, modo2 = paso_sim(fsm2, val2, kal2, dist2, modo2, "GUIDED", True, 95.0, 12.0, t2, dt_medidas=0.033)
        assert "Rechazo MRE" in mot2 and z2 == z2_ref, "UT-13c"
        return "UT-13 [Aislada]: Outliers no actualizan Kalman; outlier en control lleva a seguridad"

    def ut_14():
        fsm, val, kal = nuevo_contexto()
        dist_ant, modo_ant, t = None, "GUIDED", 0.0
        for _ in range(10):
            t += 0.033
            est, dist_ant, _, _, _, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 100.0, 0.0, t, dt_medidas=0.033)
        t += 0.033
        est, dist_ant, _, _, ord_m, mot, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", False, None, 0.0, t, dt_medidas=0.033)
        assert est == EstadoVuelo.EMERGENCIA and ord_m == 0.0 and mot == "FALLO_CAMARA", "UT-14"
        return (
            "UT-14 [Aislada]: simulación de fallo de cámara en paso_sim; "
            "no verifica la ruta real de main(), el registro ni la liberación de recursos"
        )

    def ut_15():
        fsm, val, kal = nuevo_contexto()
        dist_ant, modo_ant, t = None, "GUIDED", 0.0
        for _ in range(10):
            t += 0.033
            est, dist_ant, _, _, _, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 100.0, 0.0, t, dt_medidas=0.033)
        assert est == EstadoVuelo.CONTROL_ACTIVO, "UT-15 setup"
        
        t += 0.033
        est, dist_ant, _, _, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "ALT_HOLD", True, 100.0, 0.0, t, dt_medidas=0.033)
        assert est == EstadoVuelo.EMERGENCIA and fsm.en_emergencia_latched and ord_m is None, "UT-15d"
        rearmado = fsm.rearmar_sistema_supervisado(
            autorizacion_operador=True, enlace_verificado=True, vehiculo_desarmado=True,
            modo_verificado=True, calibracion_autorizada=True, transporte_verificado=True
        )
        assert rearmado
        t += 0.033
        est, dist_ant, _, _, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 100.0, 0.0, t, dt_medidas=0.033)
        assert est == EstadoVuelo.VALIDACION and fsm.frames_consecutivos == 1 and ord_m is None, "UT-15e"
        return "UT-15 [Aislada]: cambio de modo durante control enclava seguridad; rearme y revalidación requeridos"

    def ut_15f():
        """Integra visión sintética válida con fallo al enviar una orden no nula."""
        fsm_t, val_t, kal_t = nuevo_contexto()
        t_now = time.monotonic()
        fsm_t.estado = EstadoVuelo.CONTROL_ACTIVO
        fsm_t.frames_consecutivos = FRAMES_VALIDACION_REQ
        fsm_t.ultimo_tiempo_visto = t_now - 0.033
        pts_img, _ = cv2.projectPoints(
            pts_3d, np.zeros((3, 1), dtype=np.float32),
            np.array([[0.0], [0.0], [100.0]], dtype=np.float32), mtx_t, dist_t
        )
        esquinas_sinteticas = pts_img.reshape(1, 4, 2).astype(np.float32)

        class DetectorSintetico:
            def detectMarkers(self, gray):
                return [esquinas_sinteticas], np.array([[7]], dtype=np.int32), []

        llamadas = []
        def mock_transmisor_roto(veh, vx, vy, vz):
            llamadas.append((vx, vy, vz))
            if any(abs(float(v)) > 1e-12 for v in (vx, vy, vz)):
                raise RuntimeError("Fallo de bus MAVLink simulado para orden no nula")
            return True  # La compensación cero sí se acepta en el mock; no simula parada física.

        resultado = ejecutar_ciclo_control_integrado(
            fsm=fsm_t, kalman=kal_t, validador=val_t,
            mtx=MTX_RUNCAM_CALIBRADA, dist=DIST_RUNCAM_CALIBRADA,
            puntos_3d=pts_3d, frame=np.zeros((480, 640, 3), dtype=np.uint8),
            modo_pixhawk="GUIDED", modo_anterior="GUIDED", t_actual=t_now,
            calibracion_autorizada_ok=True, distancia_anterior_in=100.0,
            dt_medidas_in=0.033, transmisor_mavlink=mock_transmisor_roto,
            vehiculo_ref=None, detector=DetectorSintetico(), id_objetivo=7,
        )
        assert resultado["z_candidata"] is not None, "UT-15f-a: no se ejercitó la cadena visual"
        assert resultado["orden_mavlink_solicitada"] is not None
        assert abs(resultado["orden_mavlink_solicitada"]) > 0.0, "UT-15f-b: no se envió orden no nula"
        assert resultado["ok_envio"] is False, "UT-15f-c"
        assert "EXCEPCION_TRANSPORTE" in resultado["estado_transporte"], "UT-15f-d"
        assert fsm_t.en_fallo_transporte_latched, "UT-15f-e"
        assert resultado["estado_final"] == EstadoVuelo.EMERGENCIA, "UT-15f-f"
        assert resultado["orden_mavlink"] == 0.0, "UT-15f-g"
        assert resultado["resultado_transporte"] == "ERROR_TRANSPORTE_ACTUAL", "UT-15f-h"
        assert len(llamadas) == 2 and llamadas[1] == (0.0, 0.0, 0.0), "UT-15f-i: no se intentó cero compensatorio"
        assert resultado["ok_compensacion"] is True, "UT-15f-j: mock no aceptó compensación cero"
        fsm_t.reiniciar_a_busqueda()
        assert fsm_t.en_fallo_transporte_latched and fsm_t.estado == EstadoVuelo.EMERGENCIA
        return (
            "UT-15f [Integración parcial]: fallo de transmisión simulado, "
            "enclavamiento activado e intento de orden cero aceptado por mock; "
            "parada física no verificada"
        )
    
    
    def ut_mascara_mavlink_velocidad_xy():
        mask = MAVLINK_TYPE_MASK_VELOCIDAD_XY
        # Posición ignorada, Vz ignorada, aceleración ignorada, FORCE_SET apagado,
        # yaw/yaw-rate ignorados; Vx/Vy quedan activos.
        assert mask & 0b0000000000000111 == 0b0000000000000111
        assert mask & (1 << 5)
        assert mask & 0b0000000111000000 == 0b0000000111000000
        assert (mask & (1 << 9)) == 0
        assert mask & (1 << 10) and mask & (1 << 11)
        assert (mask & ((1 << 3) | (1 << 4))) == 0
        assert (mask & (1 << 12)) == 0
        return "UT-MAVLink-mask: Vx/Vy activos, Vz/posiciones/aceleración/yaw ignorados y FORCE_SET apagado"

    def ut_cierre_recursos_excepciones():
        mensajes = []
        class RecursoMock:
            def close(self):
                raise OSError("fallo simulado de cierre")
        ok = ejecutar_cierre_seguro("MOCK", RecursoMock().close, informar=mensajes.append)
        assert ok is False
        assert len(mensajes) == 1 and "OSError" in mensajes[0]
        mensajes.clear()
        ok = ejecutar_cierre_seguro("MOCK_OK", lambda: None, informar=mensajes.append)
        assert ok is True and "OK" in mensajes[0]
        return "UT-cierre-recursos: excepciones simuladas reportadas y cierre continúa"

    def ut_pasarela_bloquea_movimiento_enclavado():
        fsm_t, _, _ = nuevo_contexto()
        fsm_t.forzar_emergencia(transporte_roto=True)
        llamadas = []
        def tx_mock(veh, vx, vy, vz):
            llamadas.append((vx, vy, vz))
        ok_mov, estado_mov = despachar_orden_mavlink(tx_mock, object(), 0.12, fsm=fsm_t)
        assert not ok_mov and "BLOQUEADO" in estado_mov
        assert llamadas == [], "Se transmitió movimiento con enclavamiento activo"
        ok_cero, _ = despachar_orden_mavlink(tx_mock, object(), 0.0, fsm=fsm_t)
        assert ok_cero and llamadas == [(0.0, 0.0, 0.0)]
        return "UT-enclavamiento-pasarela: bloquea movimiento no nulo y permite solo orden cero"

    def ut_rearme_no_borra_fallo_transporte():
        fsm_t, _, _ = nuevo_contexto()
        fsm_t.forzar_emergencia(transporte_roto=True)
        ok = fsm_t.rearmar_sistema_supervisado(
            autorizacion_operador=True, enlace_verificado=True,
            vehiculo_desarmado=True, modo_verificado=True,
            calibracion_autorizada=True, transporte_verificado=True)
        assert not ok
        assert fsm_t.en_fallo_transporte_latched
        assert fsm_t.en_emergencia_latched
        return "UT-rearme-transporte: fallo de transporte exige reinicio/reconexión supervisados"

    def ut_rearme_negativo():
        fsm, _, _ = nuevo_contexto()
        fsm.forzar_emergencia(transporte_roto=True)
        rearmado_fallido = fsm.rearmar_sistema_supervisado(
            autorizacion_operador=False,
            enlace_verificado=True,
            vehiculo_desarmado=True,
            modo_verificado=True,
            calibracion_autorizada=True,
            transporte_verificado=True,
        )
        assert not rearmado_fallido, "UT-rearme-negativo-a"
        assert fsm.en_fallo_transporte_latched, "UT-rearme-negativo-b"
        return "UT-rearme-negativo [Aislada]: Rechazo de rearme sin autorización explícita"

    def ut_seleccion_id_objetivo():
        esquinas = [
            np.full((1, 4, 2), 10.0, dtype=np.float32),
            np.full((1, 4, 2), 20.0, dtype=np.float32),
        ]
        ids = np.array([[3], [7]], dtype=np.int32)
        seleccion, indice, motivo = seleccionar_marcador_objetivo(
            esquinas, ids, 7
        )
        assert motivo == "OK"
        assert indice == 1
        assert np.all(seleccion == 20.0)
        seleccion, indice, motivo = seleccionar_marcador_objetivo(
            esquinas, ids, 9
        )
        assert seleccion is None
        assert indice is None
        assert "ID_OBJETIVO_NO_DETECTADO" in motivo
        return "Selección determinista del ID ArUco objetivo"

    def ut_integridad_calibracion_y_dt():
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ok, motivo = validar_frame_y_calibracion(
            frame, True, mtx=MTX_RUNCAM_CALIBRADA, dist=DIST_RUNCAM_CALIBRADA
        )
        assert ok and motivo == "OK"
        mtx_distinta = MTX_RUNCAM_CALIBRADA.copy()
        mtx_distinta[0, 0] += 25.0
        ok, motivo = validar_frame_y_calibracion(
            frame, True, mtx=mtx_distinta, dist=DIST_RUNCAM_CALIBRADA
        )
        assert ok and motivo == "OK", "Una calibración local válida no debe compararse ni sustituirse por matrices embebidas"
        _, _, kal = nuevo_contexto()
        fallo_dt = False
        try:
            kal.actualizar(100.0, dt_medidas=0.75)
        except ValueError:
            fallo_dt = True
        assert fallo_dt, "Debe rechazarse dt largo en el filtro Kalman"
        fallo_dt_limite = False
        try:
            kal.actualizar(100.0, dt_medidas=MAX_DT_CICLO_S)
        except ValueError:
            fallo_dt_limite = True
        assert fallo_dt_limite, "Debe rechazarse dt exactamente igual al límite seguro"
        return "UT-integridad [Aislada]: matrices ópticas reales y dt >= límite rechazados"

    def ut_16():
        _, _, npz_disco, cal_aut, _ = cargar_calibracion_optica(
            rutas_busqueda=["ruta_inexistente_aud05.npz"], exigir_npz=True)
        assert (not npz_disco) and (not cal_aut), "UT-16a"
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            ruta_tmp = tmp.name
        try:
            mtx_falsa = np.array([[np.nan, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
            np.savez(ruta_tmp, mtx=mtx_falsa, dist=DIST_RUNCAM_CALIBRADA)
            _, _, _, cal_falsa, _ = cargar_calibracion_optica(rutas_busqueda=[ruta_tmp], exigir_npz=True)
            assert not cal_falsa, "UT-16b"

            mtx_local = MTX_RUNCAM_CALIBRADA.copy()
            mtx_local[0, 0] += 5.0  # Calibración local distinta pero estructuralmente válida.
            np.savez(ruta_tmp, mtx=mtx_local, dist=DIST_RUNCAM_CALIBRADA)
            mtx_cargada, _, npz_ok, cal_ok, _ = cargar_calibracion_optica(rutas_busqueda=[ruta_tmp], exigir_npz=True)
            assert npz_ok and cal_ok and abs(float(mtx_cargada[0, 0]) - float(mtx_local[0, 0])) < 1e-9, "UT-16c"
        finally:
            if os.path.exists(ruta_tmp):
                os.remove(ruta_tmp)
        frame_720p = np.zeros((720, 1280, 3), dtype=np.uint8)
        ok_720, _ = validar_frame_y_calibracion(frame_720p, True)
        assert not ok_720, "UT-16d"
        return "UT-16 [Aislada]: Política estricta de .npz y resolución 640x480"

    def ut_17():
        fsm, val, kal = nuevo_contexto()
        dist_ant, modo_ant, t = None, "GUIDED", 0.0
        for _ in range(10):
            t += 0.033
            est, dist_ant, _, _, ord_m, _, modo_ant = paso_sim(fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 100.0, 0.0, t, dt_medidas=0.033)
        assert est == EstadoVuelo.CONTROL_ACTIVO and ord_m > 0.0, "UT-17 previa"
        t += 0.600
        est, dist_ant, _, _, ord_m_bloqueo, _, modo_ant = paso_sim(
            fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 100.0, 0.0, t, dt_medidas=0.600)
        assert est == EstadoVuelo.EMERGENCIA and ord_m_bloqueo == 0.0 and dist_ant is None, "UT-17a"
        rearme_ok = fsm.rearmar_sistema_supervisado(autorizacion_operador=True, enlace_verificado=True, vehiculo_desarmado=True, modo_verificado=True, calibracion_autorizada=True, transporte_verificado=True)
        assert rearme_ok is True, "UT-17-rearme: rearme válido rechazado"
        t += 0.033
        est, dist_ant, _, _, ord_m_reval, _, modo_ant = paso_sim(
            fsm, val, kal, dist_ant, modo_ant, "GUIDED", True, 100.0, 0.0, t, dt_medidas=0.033)
        assert est == EstadoVuelo.VALIDACION and fsm.frames_consecutivos == 1 and ord_m_reval is None, "UT-17b"
        return "UT-17 [Aislada]: Watchdog en CONTROL_ACTIVO ante pausa dt >= 0.5 s"

    def ut_18():
        _, _, kal = nuevo_contexto()
        f_inicial = kal.F[0, 1]
        assert abs(f_inicial - 0.033) < 1e-5, "UT-18a"
        
        dt_custom = 0.125
        kal.actualizar(110.0, dt_medidas=dt_custom)
        
        assert abs(kal.F[0, 1] - dt_custom) < 1e-5, "UT-18b"
        assert abs(kal.Q[0, 0] - (0.1 * dt_custom)) < 1e-5, "UT-18c"
        assert np.allclose(kal.P, kal.P.T, atol=1e-7), "UT-18d: Matriz P perdió simetría"
        vals_aut = np.linalg.eigvalsh(kal.P)
        assert np.all(vals_aut >= -1e-12), "UT-18e: Matriz P dejó de ser semidefinida positiva"
        return "UT-18 [Aislada]: Verificación formal de actualización de F, Q y estabilidad numérica de P (Joseph)"



    def ut_dt_limite_ciclo_integrado():
        fsm, val, kal = nuevo_contexto()
        llamadas = []

        def mock_transmisor(veh, vx, vy, vz):
            llamadas.append((vx, vy, vz))

        resultado = ejecutar_ciclo_control_integrado(
            fsm=fsm,
            kalman=kal,
            validador=val,
            mtx=MTX_RUNCAM_CALIBRADA,
            dist=DIST_RUNCAM_CALIBRADA,
            puntos_3d=pts_3d,
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            modo_pixhawk="GUIDED",
            modo_anterior="GUIDED",
            t_actual=time.monotonic(),
            calibracion_autorizada_ok=True,
            distancia_anterior_in=100.0,
            dt_medidas_in=MAX_DT_CICLO_S,
            transmisor_mavlink=mock_transmisor,
            vehiculo_ref=None,
            detector=None,
        )

        assert resultado["estado_final"] == EstadoVuelo.EMERGENCIA
        assert resultado["orden_mavlink"] == 0.0
        assert resultado["motivo_rechazo"].startswith("PAUSA_CICLO_EXCESIVA")
        assert llamadas and llamadas[0] == (0.0, 0.0, 0.0)

        return (
            "UT-dt-integrado: dt exactamente en el límite provoca "
            "EMERGENCIA e intento de orden cero"
        )


    def ut_frame_estricto():
        frame_ok = np.zeros((480, 640, 3), dtype=np.uint8)
        assert validar_frame_y_calibracion(
            frame_ok, True, mtx=MTX_RUNCAM_CALIBRADA, dist=DIST_RUNCAM_CALIBRADA
        )[0]
        casos = [
            (None, "FRAME_NO_ES_NUMPY_NDARRAY"),
            (np.zeros((480, 640), dtype=np.uint8), "FRAME_DIMENSIONES_INVALIDAS"),
            (np.zeros((480, 640, 4), dtype=np.uint8), "FRAME_NO_ES_BGR_DE_TRES_CANALES"),
            (np.zeros((480, 640, 3), dtype=object), "FRAME_DTYPE_NO_COMPATIBLE"),
        ]
        for frame, esperado in casos:
            ok, motivo = validar_frame_y_calibracion(
                frame, True, mtx=MTX_RUNCAM_CALIBRADA, dist=DIST_RUNCAM_CALIBRADA
            )
            assert not ok and esperado in motivo, (esperado, motivo)
        return "UT-frame-estricto: ndarray, dimensiones, canales BGR y dtype"

    def ut_dt_invalidos_no_vision():
        for dt in (None, float("nan"), float("inf"), 0.0, -0.1, True, "0.03"):
            fsm_t, val_t, kal_t = nuevo_contexto()
            llamadas = []
            class DetectorNoDebeEjecutarse:
                def detectMarkers(self, gray):
                    llamadas.append("vision")
                    raise AssertionError("No debe ejecutarse visión con dt inválido")
            resultado = ejecutar_ciclo_control_integrado(
                fsm=fsm_t, kalman=kal_t, validador=val_t,
                mtx=MTX_RUNCAM_CALIBRADA, dist=DIST_RUNCAM_CALIBRADA,
                puntos_3d=pts_3d, frame=np.zeros((480, 640, 3), dtype=np.uint8),
                modo_pixhawk="GUIDED", modo_anterior="GUIDED", t_actual=1.0,
                calibracion_autorizada_ok=True, distancia_anterior_in=None,
                dt_medidas_in=dt, transmisor_mavlink=lambda *args: True,
                vehiculo_ref=None, detector=DetectorNoDebeEjecutarse(), id_objetivo=7,
            )
            assert resultado["estado_final"] == EstadoVuelo.EMERGENCIA
            assert "DT_INVALIDO" in resultado["motivo_rechazo"]
            assert not llamadas
        return "UT-dt-invalidos: NaN/inf/cero/negativo/no numérico/bool bloquean visión"

    def ut_pose_externa_transformacion():
        # Marcador frontal vertical a 2 m; rotación cámara-marcador ideal diag(1,-1,-1).
        R_cm = np.diag([1.0, -1.0, -1.0])
        rvec_test, _ = cv2.Rodrigues(R_cm)
        pose = calcular_pose_externa_ned(rvec_test, np.array([[0.0], [0.0], [200.0]]), 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        assert abs(pose["x_ned_m"] - 2.0) < 1e-5, pose
        assert abs(pose["y_ned_m"]) < 1e-5, pose
        assert abs(pose["z_ned_m"] + 1.5) < 1e-5, pose
        assert pose["frame"] == "LOCAL_NED" and pose["units"] == "m_rad"
        assert abs(abs(pose["yaw_rad"]) - np.pi) < 1e-5, pose
        return "UT-VISION-POSE: solvePnP cm->m, transformación pared/NED y orientación sintética"

    def ut_id_objetivo_none_rechazado():
        esquinas = [np.zeros((1, 4, 2), dtype=np.float32)]
        ids = np.array([[0]], dtype=np.int32)
        seleccion, indice, motivo = seleccionar_marcador_objetivo(esquinas, ids, None)
        assert seleccion is None and indice is None and motivo == "ID_OBJETIVO_NO_CONFIGURADO"
        seleccion, indice, motivo = seleccionar_marcador_objetivo(esquinas, ids, 0)
        assert seleccion is not None and indice == 0 and motivo == "OK"
        return "UT-ARUCO-ID: ID None bloqueado y solo se acepta el ID explícito 0"

    def ut_bloqueo_hasta_ekf_valido():
        fsm = FSM_Seguridad()
        t = 0.0
        for i in range(FRAMES_VALIDACION_REQ):
            t += 0.033
            estado = fsm.actualizar(True, t, navegacion_ekf_valida=False)
        assert estado == EstadoVuelo.VALIDACION and fsm.frames_consecutivos == FRAMES_VALIDACION_REQ
        t += 0.033
        estado = fsm.actualizar(True, t, navegacion_ekf_valida=True)
        assert estado == EstadoVuelo.CONTROL_ACTIVO
        return "UT-EKF-GATE: diez frames no autorizan control hasta que EKF/fuente externa sean válidos"

    def ut_publicador_vision_mavlink_mock():
        global DRONEKIT_DISPONIBLE, mavutil
        dronekit_original = DRONEKIT_DISPONIBLE
        mavutil_original = globals().get("mavutil", None)
        class Mensaje:
            def __init__(self, *args):
                self.args = args
        class Dialecto:
            MAVLink_vision_position_estimate_message = Mensaje
            MAVLink_vision_speed_estimate_message = Mensaje
        class VehiculoMock:
            def __init__(self):
                self.mensajes = []
                self.flushes = 0
            def send_mavlink(self, msg):
                self.mensajes.append(msg)
            def flush(self):
                self.flushes += 1
        try:
            DRONEKIT_DISPONIBLE = True
            class UtilMock:
                mavlink = Dialecto()
            mavutil = UtilMock()
            pub = PublicadorVisionMavlink()
            veh = VehiculoMock()
            p1 = {"x_ned_m": 1.0, "y_ned_m": 2.0, "z_ned_m": -1.5,
                  "roll_rad": 0.0, "pitch_rad": 0.0, "yaw_rad": 0.2}
            ok1, _ = pub.publicar(veh, p1, t_monotonic=1.0)
            assert ok1 and len(veh.mensajes) == 1
            p2 = dict(p1); p2["x_ned_m"] = 1.04
            ok2, _ = pub.publicar(veh, p2, t_monotonic=1.04)
            assert ok2 and len(veh.mensajes) == 3 and veh.flushes == 2
            try:
                pub.publicar(veh, p2, t_monotonic=1.30)
                raise AssertionError("Debía rechazar una cadencia externa demasiado lenta")
            except RuntimeError as exc:
                assert "FRECUENCIA_VISION_INSUFICIENTE" in str(exc)
            return "UT-VISION-TX-MOCK: separa pose/velocidad, calcula velocidad entre poses y bloquea cadencia insuficiente"
        finally:
            DRONEKIT_DISPONIBLE = dronekit_original
            if mavutil_original is None:
                globals().pop("mavutil", None)
            else:
                mavutil = mavutil_original

    pruebas = [
        ut_pose_externa_transformacion, ut_id_objetivo_none_rechazado, ut_bloqueo_hasta_ekf_valido, ut_publicador_vision_mavlink_mock,
        ut_01, ut_02, ut_03, ut_04, ut_05, ut_06, ut_07, ut_08, ut_09,
        ut_10, ut_11, ut_12, ut_13, ut_14, ut_15, ut_15f, ut_mascara_mavlink_velocidad_xy, ut_cierre_recursos_excepciones, ut_pasarela_bloquea_movimiento_enclavado, ut_rearme_no_borra_fallo_transporte, ut_rearme_negativo,
        ut_frame_estricto, ut_dt_invalidos_no_vision, ut_seleccion_id_objetivo, ut_integridad_calibracion_y_dt, ut_16, ut_17, ut_18, ut_dt_limite_ciclo_integrado
    ]

    for fn_prueba in pruebas:
        descripcion = fn_prueba()
        print(f"  [PASS] {descripcion}")

    print("=" * 80)
    print(f"RESULTADO DE EJECUCIÓN EN CONSOLA: {len(pruebas)}/{len(pruebas)} FUNCIONES DE PRUEBA SUPERADAS")
    print("=" * 80)


def ejecutar_cierre_seguro(nombre, accion, informar=print):
    """Ejecuta una operación de cierre aislada y comunica su resultado sin ocultar excepciones."""
    try:
        accion()
        informar(f"[CIERRE] {nombre}: OK (llamada completada; recurso físico no verificado)")
        return True
    except Exception as exc:
        informar(f"[CIERRE] {nombre}: ERROR {type(exc).__name__}: {exc}")
        return False


# ==========================================
# 9. BUCLE PRINCIPAL DE VUELO Y DRY-RUN
# ==========================================
def main():
    if os.environ.get("AUD05_HABILITAR_VUELO_AUTONOMO") != "1":
        print("[BLOQUEO DE SEGURIDAD] El vuelo autónomo está deshabilitado por defecto.")
        print("La ejecución está bloqueada por defecto. No active esta variable para saltarse la validación; requiere checklist, SITL y banco.")
        print("Esta variable no certifica seguridad ni sustituye los enclavamientos físicos y la supervisión del piloto.")
        return
    if os.environ.get("AUD05_MAVLINK_CONFIG_VALIDADA") != "1":
        print("[BLOQUEO DE SEGURIDAD] Falta AUD05_MAVLINK_CONFIG_VALIDADA=1.")
        print("No se inicia el control hasta validar type_mask, frame y ejes con el firmware/vehículo exactos en SITL o banco.")
        print("No establezca esta variable solo para saltarse el bloqueo: requiere evidencia de la prueba de integración.")
        return
    requisitos_previos = {
        "AUD05_VISION_EKF_VALIDADA": "1",
        "AUD05_MARCO_NED_VALIDADO": "1",
        "AUD05_EXTRINSECOS_CAMARA_VALIDADOS": "1",
        "AUD05_EKF_FUSION_CONFIRMADA": "1",
    }
    ausentes = [k for k, v in requisitos_previos.items() if os.environ.get(k) != v]
    if ausentes:
        print("[BLOQUEO DE SEGURIDAD] Falta evidencia de integración externa: " + ", ".join(ausentes))
        print("No se inicia el bucle de control. Estas confirmaciones requieren pruebas SITL/banco y revisión de Mission Planner.")
        return
    try:
        rumbo_normal_pared_deg, offset_camara_cuerpo_m, origen_ekf_en_pared_m = leer_configuracion_geometrica_entorno()
    except ValueError as exc:
        print(f"[BLOQUEO DE SEGURIDAD] Geometría externa no configurada: {exc}")
        return
    if not DRONEKIT_DISPONIBLE:
        print("[ERROR CRÍTICO] DroneKit / pymavlink no están instaladas en el entorno. Imposible conectar con Pixhawk.")
        sys.exit(1)

    print("Iniciando AUD-05 v5.22 (candidata de integración de visión externa; no certificada para vuelo)...")
    
    vehiculo = None
    cap = None
    archivo_log = None
    fsm = None

    try:
        vehiculo = connect('udpin:0.0.0.0:14550', wait_ready=True)

        mtx, dist, npz_desde_disco, cal_matriz_ok, motivo_cal = cargar_calibracion_optica()
        print(f"[CALIBRACIÓN] Estado: {motivo_cal} | Autorizada para GUIDED: {cal_matriz_ok}")
        if not (npz_desde_disco and cal_matriz_ok):
            raise RuntimeError("calibracion_runcam.npz ausente o inválido; se aborta sin usar calibración de respaldo.")

        detector = crear_detector_aruco()

        puntos_3d_aruco = np.array([
            [-LADO_REAL_CM / 2,  LADO_REAL_CM / 2, 0],
            [ LADO_REAL_CM / 2,  LADO_REAL_CM / 2, 0],
            [ LADO_REAL_CM / 2, -LADO_REAL_CM / 2, 0],
            [-LADO_REAL_CM / 2, -LADO_REAL_CM / 2, 0]
        ], dtype=np.float32)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir el dispositivo de captura de video (cámara USB).")
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUCION_CALIBRADA[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUCION_CALIBRADA[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        kalman = FiltroKalmanCinematico(dt_inicial=0.033, var_pos=0.1, var_vel=0.05, var_medida=8.0, pos_inicial=120.0)
        validador = ValidadorVisual()
        fsm = FSM_Seguridad()
        publicador_vision = PublicadorVisionMavlink()

        distancia_anterior = None
        distancia_filtrada = kalman.X[0, 0]
        velocidad_segura = 0.0
        t_inicio_mono = time.monotonic()
        t_ultimo_ciclo = time.monotonic()
        t_ultima_medida_valida = None
        modo_anterior = vehiculo.mode.name if vehiculo.mode is not None else "DESCONOCIDO"

        ruta_log = os.path.join(
            os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".",
            "registro_failsafe_dryrun.txt"
        )
        archivo_log = open(ruta_log, "a", buffering=1, encoding="utf-8")
        archivo_log.write(
            f"\n--- INICIO SESIÓN {time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"NPZ_Disco={npz_desde_disco} | Cal_Autorizada={cal_matriz_ok} ({motivo_cal}) ---\n"
        )

        while True:
            ret, frame = cap.read()
            t_actual = time.monotonic()
            t_rel = t_actual - t_inicio_mono
            dt_ciclo = max(0.001, t_actual - t_ultimo_ciclo)
            t_ultimo_ciclo = t_actual
            
            modo_pixhawk = vehiculo.mode.name if vehiculo.mode is not None else "DESCONOCIDO"

            if hay_transicion_guided(modo_anterior, modo_pixhawk):
                t_ultima_medida_valida = None

            if not ret or frame is None:
                estado_previo = fsm.estado
                estado_actual = fsm.forzar_emergencia(transporte_roto=False)
                distancia_anterior = None
                velocidad_segura = 0.0
                vel_env_str = "NO_ENV"
                estado_env = "NO_GUIDED"
                if modo_pixhawk == "GUIDED":
                    ok_env, estado_env = despachar_orden_mavlink(enviar_velocidad_mavlink, vehiculo, 0.0, fsm=fsm)
                    if not ok_env:
                        fsm.forzar_emergencia(transporte_roto=True)
                        ok_comp, estado_comp = despachar_orden_mavlink(
                            enviar_velocidad_mavlink, vehiculo, 0.0, fsm=fsm
                        )
                        estado_env = f"{estado_env}; COMPENSACION={estado_comp}"
                        vel_env_str = "ERR_ENV"
                    else:
                        vel_env_str = "+0.00m/s (ORDEN ENVIADA)"
                texto_err = (
                    f"T: {t_rel:7.3f}s | FSM: {estado_actual.name:18s} | "
                    f"Trans: {estado_previo.name}->{estado_actual.name} | Modo: {modo_pixhawk:10s} | "
                    f"Z_Cand:   None | Z_Est: {kalman.X[0,0]:6.1f}cm | VelZ: +0.00 m/s | "
                    f"Env: {vel_env_str:8s} | ResultadoTx: FALLO_CAPTURA_CAMARA | Filtro: FALLO_CAMARA_CRITICO (cap.read False)\n"
                )
                print(texto_err.strip())
                if archivo_log:
                    archivo_log.write(texto_err)
                break

            # Invocación centralizada del ciclo de control integrado unificado
            res_ciclo = ejecutar_ciclo_control_integrado(
                fsm=fsm, kalman=kalman, validador=validador, mtx=mtx, dist=dist,
                puntos_3d=puntos_3d_aruco, frame=frame, modo_pixhawk=modo_pixhawk,
                modo_anterior=modo_anterior, t_actual=t_actual, calibracion_autorizada_ok=cal_matriz_ok,
                distancia_anterior_in=distancia_anterior, dt_medidas_in=dt_ciclo,
                transmisor_mavlink=enviar_velocidad_mavlink, vehiculo_ref=vehiculo, detector=detector,
                id_objetivo=ID_ARUCO_OBJETIVO,
                rumbo_normal_pared_deg=rumbo_normal_pared_deg,
                offset_camara_cuerpo_m=offset_camara_cuerpo_m,
                origen_ekf_en_pared_m=origen_ekf_en_pared_m,
                publicador_vision=publicador_vision,
                navegacion_ekf_valida=(verificar_configuracion_ekf_externa(vehiculo, exigir_confirmacion=True)[0] and modo_pixhawk == "GUIDED" and publicador_vision.poses_publicadas >= FRAMES_VALIDACION_REQ)
            )

            estado_actual = res_ciclo["estado_final"]
            distancia_anterior = res_ciclo["distancia_anterior"]
            distancia_filtrada = res_ciclo["distancia_filtrada"]
            velocidad_segura = res_ciclo["velocidad_segura"]
            orden_mavlink = res_ciclo["orden_mavlink"]
            ok_envio = res_ciclo["ok_envio"]
            estado_transporte = res_ciclo["estado_transporte"]
            resultado_transporte = res_ciclo["resultado_transporte"]
            z_candidata = res_ciclo["z_candidata"]
            motivo_rechazo = res_ciclo["motivo_rechazo"]
            if z_candidata is not None:
                t_ultima_medida_valida = t_actual
            elif publicador_vision is not None:
                publicador_vision.invalidar_historial()

            modo_anterior = modo_pixhawk

            # Construcción de la interfaz y traza con estado sincronizado
            if orden_mavlink is None:
                estado_modo = f"{modo_pixhawk} (Piloto Manual)"
                vel_env_str = "NO_ENV"
            elif resultado_transporte == "ERROR_TRANSPORTE_ACTUAL":
                estado_modo = f"ERROR_ENVIO ({estado_transporte})"
                vel_env_str = "ERR_ENV"
            elif resultado_transporte == "BLOQUEADO_POR_FALLO_PREVIO" or fsm.en_fallo_transporte_latched:
                estado_modo = "GUIDED (ENCLAVAMIENTO ACTIVO)"
                vel_env_str = "BLOQUEADO"
            else:
                estado_modo = (
                    "GUIDED (Control Autónomo)"
                    if (
                        fsm.referencia_visual_validada
                        and cal_matriz_ok
                        and estado_actual == EstadoVuelo.CONTROL_ACTIVO
                    )
                    else "GUIDED (Parada Segura 0 m/s)"
                )
                vel_env_str = (
                    f"{orden_mavlink:+.2f}m/s"
                    if ok_envio
                    else "ERR_ENV"
                )

            color_estado = (
                (0, 255, 0)
                if (
                    estado_actual == EstadoVuelo.CONTROL_ACTIVO
                    and cal_matriz_ok
                    and not fsm.en_fallo_transporte_latched
                    and not fsm.en_emergencia_latched
                )
                else (0, 0, 255)
            )
            nombre_estado_str = (
                f"VALIDANDO ({fsm.frames_consecutivos}/{FRAMES_VALIDACION_REQ})"
                if estado_actual == EstadoVuelo.VALIDACION
                else estado_actual.name
            )
            z_cand_str = (
                f"{z_candidata:6.1f}cm"
                if z_candidata is not None
                else "  None  "
            )

            cv2.putText(frame, f"FSM: {nombre_estado_str}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)
            cv2.putText(frame, f"Modo: {estado_modo}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 100), 2)
            cv2.putText(frame, f"VelZ Calc: {velocidad_segura:+.2f} m/s | Env: {vel_env_str}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.putText(frame, f"Filtro: {motivo_rechazo}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Z Cand: {z_cand_str.strip()} | Z Est: {distancia_filtrada:.1f}cm", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            edad_medida_s = (
                max(0.0, t_actual - t_ultima_medida_valida)
                if t_ultima_medida_valida is not None else float("inf")
            )
            edad_medida_str = f"{edad_medida_s:.3f}s" if np.isfinite(edad_medida_s) else "N/D"
            texto_log = (
                f"T: {t_rel:7.3f}s | EdadMedida: {edad_medida_str:>7s} | FSM: {nombre_estado_str:18s} | Trans: {res_ciclo['transicion']:25s} | "
                f"Modo: {modo_pixhawk:10s} | Z_Cand: {z_cand_str} | "
                f"Z_Est: {distancia_filtrada:6.1f}cm | Vz_K: {kalman.X[1, 0]:+6.2f}cm/s | "
                f"Edad: {fsm.tiempo_ciego*1000.0:4.0f}ms | VelZ: {velocidad_segura:+.2f} m/s | "
                f"Env: {vel_env_str:8s} | ResultadoTx: {resultado_transporte:34s} | Filtro: {motivo_rechazo}\n"
            )
            print(texto_log.strip())
            if archivo_log:
                archivo_log.write(texto_log)

            cv2.imshow('AUD-05 v5.22 - Visión externa (candidata de banco)', frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord('q'):
                break
            elif tecla == ord('r'):
                # Rearme deliberado solo tras comprobaciones en vivo y vehículo desarmado.
                condiciones = verificar_condiciones_rearme(
                    vehiculo, cal_matriz_ok, modo_simulacion=False
                )
                condiciones["calibracion_autorizada"] = bool(
                    cal_matriz_ok and verificar_integridad_calibracion(mtx, dist, verificar_correspondencia_runcam=False)
                )
                condiciones["transporte_verificado"] = bool(
                    condiciones["enlace_verificado"] and
                    vehiculo is not None and
                    not fsm.en_fallo_transporte_latched
                )
                rearme_ok = fsm.rearmar_sistema_supervisado(
                    autorizacion_operador=True,  # La tecla 'r' es la acción explícita del operador.
                    enlace_verificado=condiciones["enlace_verificado"],
                    vehiculo_desarmado=condiciones["vehiculo_desarmado"],
                    modo_verificado=condiciones["modo_verificado"],
                    calibracion_autorizada=condiciones["calibracion_autorizada"],
                    transporte_verificado=condiciones["transporte_verificado"]
                )
                estado_rearme = "ACEPTADO" if rearme_ok else "RECHAZADO"
                print(
                    f"[REARME SUPERVISADO] {estado_rearme} | "
                    f"Condiciones: {condiciones}"
                )
                if archivo_log:
                    archivo_log.write(
                        f"[REARME SUPERVISADO] {estado_rearme} | {condiciones}\n"
                    )

    
    except Exception as e:
        # Una excepción inesperada debe activar el enclavamiento
        # de emergencia antes de iniciar la secuencia de cierre.
        if fsm is not None:
            try:
                fsm.forzar_emergencia(transporte_roto=False)
            except Exception as exc_fsm:
                print(
                    "[ERROR CRÍTICO] No se pudo activar la emergencia: "
                    f"{type(exc_fsm).__name__}: {exc_fsm}"
                )

        mensaje_error = (
            f"[EXCEPCIÓN CRÍTICA EN BUCLE PRINCIPAL] "
            f"{type(e).__name__}: {e}"
        )
        print(mensaje_error)

        if archivo_log is not None:
            try:
                archivo_log.write(mensaje_error + "\n")
                archivo_log.flush()
            except Exception as exc_log:
                print(
                    "[ERROR] No se pudo registrar la excepción: "
                    f"{type(exc_log).__name__}: {exc_log}"
                )

    finally:

        try:
            if DRONEKIT_DISPONIBLE and vehiculo is not None and vehiculo.mode is not None and vehiculo.mode.name == "GUIDED":
                ok_parada, estado_parada = despachar_orden_mavlink(
                    enviar_velocidad_mavlink, vehiculo, 0.0, fsm=fsm
                )
                if not ok_parada:
                    if fsm is not None:
                        fsm.forzar_emergencia(transporte_roto=True)
                    print(f"[FALLO CRÍTICO] No se confirmó el envío de orden cero al cerrar: {estado_parada}")
                    if archivo_log is not None:
                        archivo_log.write(
                            f"[CIERRE] FALLO_ENVIO_ORDEN_CERO | {estado_parada} | "
                            f"Parada física NO confirmada\n"
                        )
                else:
                    print("[CIERRE] Orden cero enviada; esto no confirma por sí solo la parada física.")
                    if archivo_log is not None:
                        archivo_log.write("[CIERRE] ORDEN_CERO_ENVIADA; parada física no confirmada\n")
            else:
                print("[CIERRE] No se envió orden cero: vehículo no disponible o modo distinto de GUIDED.")
        except Exception as exc:
            if fsm is not None:
                fsm.forzar_emergencia(transporte_roto=True)
            print(f"[FALLO CRÍTICO] Excepción al intentar orden cero de cierre: {exc}")
            if archivo_log is not None:
                archivo_log.write(f"[CIERRE] EXCEPCION_ORDEN_CERO: {type(exc).__name__}: {exc}\n")
        if archivo_log is not None:
            ejecutar_cierre_seguro("REGISTRO_CLOSE", archivo_log.close)

        if cap is not None:
            ejecutar_cierre_seguro("CAMARA_RELEASE", cap.release)

        ejecutar_cierre_seguro("VENTANAS_DESTROY", cv2.destroyAllWindows)

        if DRONEKIT_DISPONIBLE and vehiculo is not None:
            ejecutar_cierre_seguro("VEHICULO_CLOSE", vehiculo.close)
        print("[CIERRE DEL SISTEMA] Secuencia de cierre finalizada; la liberación individual y la parada física no implican confirmación del estado físico del vehículo.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        ejecutar_pruebas_unitarias()
    else:
        main()