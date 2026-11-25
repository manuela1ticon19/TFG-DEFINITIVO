# 00 — Documento maestro del nuevo proyecto TFG-DRON

**Referencia de partida:** `AUD-05_v5.22_vision_externa_candidata.py`  
**Proyecto:** TFG sobre percepción visual ArUco y estudio de asistencia de posición en vuelo interior  
**Estado:** nueva etapa de desarrollo  
**Objetivo de este documento:** mantener un contexto conciso, estable y coherente para el repositorio y las herramientas de desarrollo.

---

## 1. Objetivo y alcance

Se desea desarrollar una base de software sencilla y comprensible para estudiar una arquitectura de percepción visual basada en un marcador ArUco, cámara FPV, procesamiento en Python y estimación temporal.

La intención académica es estudiar cómo la observación visual del marcador permite obtener una medida relativa y cómo una arquitectura externa puede utilizar esa información en una investigación de asistencia de posición. El desarrollo se organizará en fases: percepción y pruebas offline; integración simulada; y, si procede, experimentación física posterior bajo supervisión y procedimientos de seguridad apropiados.

Este repositorio inicia una etapa nueva. Se aprovecharán conocimientos y componentes seleccionados del código `AUD-05_v5.22_vision_externa_candidata.py`, pero no se trasladará automáticamente la arquitectura, el volumen de código ni el backlog del workspace anterior.

**No se pretende iniciar otra auditoría indefinida.** Se priorizará la construcción de entregables concretos, pruebas reproducibles y decisiones acotadas. Las revisiones deberán ser específicas a una tarea o defecto identificado.

## 2. Contexto académico y rigor

El proyecto surge tras una primera tentativa de vuelo y la revisión de los tutores. La tentativa anterior no permitió demostrar vuelo autónomo integrado. Por ello, el nuevo trabajo debe separar y documentar claramente:

- diseño conceptual;
- código implementado;
- prueba offline;
- integración en simulación;
- prueba de banco;
- experimento físico;
- conclusión realmente respaldada por evidencia.

No se presentará como demostrado un resultado que solo haya sido previsto, simulado parcialmente, analizado o propuesto.

## 3. Plataforma de referencia

### Hardware disponible o previsto

- Cuadricóptero F450.
- Autopiloto Pixhawk (identificación y configuración exactas se documentarán).
- ArduCopter; Mission Planner muestra la versión 4.5.7 en la comprobación aportada.
- Cámara FPV RunCam.
- Enlace analógico de vídeo de 5,8 GHz.
- Captura prevista de imagen a aproximadamente 30 FPS.
- Mando RC para intervención manual y experimentación.
- Marcador ArUco como referencia visual.

### Software y herramientas

- Python.
- OpenCV y detección ArUco.
- VS Code.
- Git y GitHub.
- Ubuntu/WSL como entorno de desarrollo (versión/configuración exactas por verificar y documentar).
- ArduPilot y SITL.
- Mission Planner, para configuración, supervisión y análisis de telemetría/registros.
- MAVLink y PyMAVLink como tecnologías de comunicación presentes en los antecedentes.
- Simulador visual, a seleccionar en una fase posterior.

No se asumirán versiones de bibliotecas, parámetros, rutas, frecuencias o ajustes del autopiloto como verificados hasta documentarlos en el entorno real del repositorio.

## 4. Arquitectura conceptual de la investigación

La arquitectura que se desea estudiar se describe mediante las siguientes etapas:

**A. Captura de imagen**  
La cámara RunCam proporciona vídeo FPV mediante transmisión analógica de 5,8 GHz. La captura prevista es aproximadamente de 30 FPS. La calidad real, frecuencia efectiva, latencia y pérdidas deberán medirse, no asumirse.

**B. Procesamiento visual**  
Python/OpenCV procesa los frames y detecta el marcador ArUco. La observación debe incluir, cuando corresponda, datos de detección y métricas que permitan evaluar su calidad.

**C. Validación de observaciones — diseño de tres capas**  
Se ha planteado un validador con tres criterios:

1. **Geometría:** valorar el error de reproyección medio (MRE). Se ha propuesto MRE < 8 píxeles como criterio inicial.
2. **Dimensión/rango:** se ha propuesto ignorar observaciones fuera del intervalo 20–250 cm.
3. **Consistencia cinemática:** se ha propuesto rechazar cambios de distancia de 40 cm entre frames consecutivos, equivalentes aproximadamente a 12 m/s si el intervalo es de 1/30 s.

Estos valores son **umbrales de diseño preliminares**, no límites físicos universales ni parámetros validados. Se deberán comprobar con vídeos, datos de calibración, mediciones y análisis de falsas aceptaciones/rechazos. El criterio cinemático deberá tener en cuenta el intervalo temporal real entre observaciones, no presuponer que siempre es exactamente un frame a 30 FPS.

**D. Estimación temporal**  
Se propone estudiar un filtro de Kalman cinemático de segundo orden para suavizar la observación de posición. Se ha considerado la forma de Joseph para la actualización de covarianza por sus propiedades numéricas. Su implementación, modelo, variables, ruido y comportamiento deberán verificarse con pruebas offline y casos sintéticos.

El filtro no sustituye la validación de la detección: una observación inválida no debe tratarse como una medida fiable simplemente por pasar por un filtro.

**E. Gestión de estados del software**  
Se ha planteado una máquina de estados finita (FSM) para gestionar condiciones de disponibilidad, calidad de observación, pérdida de referencia y recuperación. Los antecedentes incluyen una propuesta de requerir 10 frames estables y un watchdog de 0,5 s (15 frames nominales a 30 FPS).

Estos valores y las transiciones relacionadas con el autopiloto son propuestas de diseño que deben verificarse y documentarse. Este documento no define procedimientos operativos para habilitar motores, cambiar modos de vuelo o gestionar una emergencia real. La pérdida de detección, las transiciones de modo y la recuperación deben tratarse como cuestiones de seguridad que requieren evaluación específica y supervisión competente.

**F. Registro**  
El software de investigación deberá poder registrar observaciones, timestamps, métricas de validación, estado del procesamiento, eventos, pérdidas de detección y resultados de pruebas, de modo que las conclusiones puedan reproducirse y analizarse.

**G. Decisión/control — descripción académica**  
El planteamiento de investigación contempla un controlador proporcional externo sobre la variable relativa observada, con una expresión conceptual del tipo:

`v = Kp · (Zactual − Zobjetivo)`

y una saturación propuesta de ±0,30 m/s. La comparación P/PI/PID se estudió anteriormente en `Comparativa_LazoCerrado_P_PI_PID.py`; ese análisis se conserva como antecedente y no se repetirá automáticamente.

La integración de decisiones de control con el autopiloto real, la transmisión de consignas de vuelo, la habilitación de GUIDED, el despegue, el aterrizaje y la lógica operativa de recuperación no son objetivos de implementación de esta fase documental/offline. La futura integración se desarrollará y probará en un entorno simulado.

## 5. Conceptos del autopiloto: descripción general

El autopiloto dispone de lazos internos de control y mezcla de actuadores. De manera general, el software externo no calcula directamente las señales individuales de los motores: el autopiloto interpreta las referencias compatibles con el modo y configuración utilizados y ejecuta sus propios procesos de estimación, control y mezcla.

La descripción de una cascada de control, sus frecuencias, los PID internos, el mezclador Quad-X o la generación PWM debe considerarse una explicación conceptual. La frecuencia nominal de un proceso no implica que todas las capas funcionen exactamente a esa frecuencia. Los detalles dependen de la versión, parámetros, hardware y configuración, y se documentarán con fuentes verificables cuando se redacte el capítulo técnico.

## 6. Simulación — segunda fase

La simulación es una fase posterior, no una condición para comenzar la base de percepción.

El objetivo es conseguir, en la medida que lo permita la plataforma elegida, un escenario representativo en el que se pueda:

- visualizar un cuadricóptero;
- visualizar un entorno interior;
- visualizar un marcador ArUco en el escenario;
- disponer de una vista de cámara simulada;
- utilizar un mando o entrada de piloto en el entorno simulado;
- procesar las imágenes mediante la cadena visual del proyecto;
- estudiar la interacción entre el software externo y un autopiloto simulado;
- registrar telemetría, observaciones y eventos;
- estudiar ruido, latencia, perturbaciones, oclusión y pérdida de detección.

Se estudiarán SITL, FlightGear, Gazebo u otras alternativas solo cuando sea necesario seleccionar una plataforma. FlightGear fue sugerido en una conversación anterior por Gemini, pero no se considera decisión definitiva. La selección se basará en representatividad del escenario, visualización, integración, dificultad de instalación, mantenimiento y reproducibilidad. No se montarán varias plataformas simultáneamente sin una razón concreta.

## 7. Código de referencia y antecedentes

### Punto de partida

`AUD-05_v5.22_vision_externa_candidata.py`

Se conserva como referencia de ideas, componentes y aprendizaje acumulado. El nuevo desarrollo no debe heredar automáticamente todas sus capas, módulos o estados. Tampoco debe alterar la copia de respaldo del archivo original.

### Antecedentes que se mantienen fuera del contexto activo

- Código principal de aproximadamente 6.000 líneas y módulos relacionados.
- Los siete chats de desarrollo/revisión.
- Documento `MEJORANDO TFG`.
- Documento `TRAS REVISION CON LOS TUTORES`.
- `Comparativa_LazoCerrado_P_PI_PID.py`.
- Registros de pruebas, auditorías, incidencias y backlog anterior.

Se consultarán puntualmente cuando se necesite una información concreta. No se importará el historial completo al contexto del nuevo repositorio ni se convertirá en una lista de tareas heredada.


## 7 bis. Archivo de calibración de cámara y datos necesarios

El repositorio debe incluir, junto al código de visión, el archivo de calibración de la cámara que se utilizará en las pruebas: **`calibracion_runcam.npz`**. Este archivo contiene (según el formato utilizado por los scripts anteriores) los parámetros intrínsecos `mtx` y de distorsión `dist`. No debe sustituirse silenciosamente por una matriz estimada o valores por defecto. Si falta, el programa de análisis deberá advertirlo claramente y no presentar resultados de pose/distancia como calibrados.

- Mantener el archivo de calibración en una ubicación documentada (preferentemente `data/calibration/` o una ruta equivalente definida por el proyecto).
- Conservar una copia de respaldo del archivo original.
- Documentar cámara, resolución de calibración, método, fecha y error de reproyección si esos datos están disponibles.
- El archivo `.npz` es un dato de calibración, no código fuente; se debe versionar si no contiene información sensible y si su tamaño/uso lo permite.
- No se dará por validada la calibración únicamente porque el archivo se cargue correctamente.

## 7 ter. Registro de antecedentes: pruebas, scripts y evidencias

Este apartado resume el trabajo previo descrito en el dossier `MEJORANDO TFG(3)/(4)` y otros documentos de continuidad. Sirve para reutilizar resultados y evitar repetir ensayos sin motivo. **Los estados reflejan lo que se documentó en el dossier, no una verificación independiente de los archivos de ejecución**. Antes de citarlo como resultado definitivo en la memoria, comprobar que se conservan los scripts, logs, vídeos, gráficos o `.BIN` que lo respaldan.

| Área / prueba documentada | Código o archivo asociado | Qué se estudió / evidencia descrita | Estado / cautela |
|---|---|---|---|
| Comparación de leyes de control | `Comparativa_LazoCerrado_P_PI_PID.py`; también se documenta `verificacion_sistematica.py` en la etapa posterior | Comparación analítica/simulada entre P, PI con anti-windup y PID filtrado, con ruido y retardo en los modelos descritos | Antecedente que se conserva; no repetir automáticamente. Revisar qué versión y resultados corresponden a cada script antes de citarlos |
| Filtro de Kalman en modelo sencillo | `simulacion_kalman.py` (nombre citado en el dossier) | Ejemplo matemático con ruido y gráfica, no una simulación física completa | Simulación simplificada; no equivale a integración con autopiloto |
| Primer intento SITL de aterrizaje/aproximación | `simulador_aterrizaje.py` (nombre citado) | Modelo de pasos idealizados, descrito como insuficiente para representar ruido, retardos y FSM | Antecedente limitado; no usar como prueba de arquitectura completa |
| Comparación sistemática de controladores | `verificacion_sistematica.py` | Modelo de lazo cerrado con P, PI anti-windup y PID filtrado; el dossier describe ruido FPV y retardo | Se documentan problemas/pendientes de revisión en versiones posteriores; no asumir resultados finales ni ejecución reproducible sin revisar artefactos |
| SITL avanzado con perturbaciones | `simulacion_sitl_avanzada.py` | Se documentó una prueba con ArduCopter SITL, perturbaciones, ruido, retraso, outliers y pérdida/oclusión de marcador; se mencionan registro y vídeo | Dossier indica ejecución, pero también señala bugs y una reejecución/corrección pendiente. No tratar como validación final de la arquitectura |
| Pruebas de visión y efecto del buffering | `visión_pnp_confallo.py` y referencias a `vision_pnp.py` | Ensayo destinado a estudiar el retardo/buffer de vídeo; se describe una versión con buffer deliberadamente no limitado y otra con mitigación | Confirmar qué variante se ejecutó y qué evidencia se guardó; no atribuir automáticamente el accidente original a una causa única |
| Ensayos de banco sin hélices y análisis de controlador | `comprobacion_P_frentePIyPID.py`; registros `Reg4_CONCELO_Controlador_P_A0` y otras pruebas descritas | Lectura ArUco, filtrado y cálculo de salidas P/PI/PID; análisis de `.BIN`, telemetría y señales `ATT`, `MODE`, `RATE`, `RCOU`; pruebas con distintas orientaciones/sujeciones | El dron estuvo inmovilizado/sujeto o apoyado en banco en los ensayos descritos; no demuestra respuesta de vuelo libre ni aceptación de GUIDED |
| Ensayos de comunicación, cámara y dry-run | Código principal y scripts auxiliares del workspace previo; registros de banco mencionados en el dossier | Se describen comprobaciones de enlace, cámara, calibración, telemetría y emergencia en pruebas sin hélices | Conservar solo la evidencia específica que se pueda identificar; separar ensayo realizado, análisis de log y recomendación |
| Candidato de arquitectura de percepción | `AUD-05_v5.22_vision_externa_candidata.py` | Código candidato con validación visual, estimación, FSM, registro y componentes de comunicación según el dossier | Punto de partida elegido para la nueva etapa, pero no certificado ni validado para operar un dron real |

### Evidencias que deben conservarse, si están disponibles

- Scripts exactos que se ejecutaron, no solamente el nombre del archivo o una versión posterior.
- Registros `.txt`/`.csv`, vídeos, gráficas y archivos DataFlash `.BIN` correspondientes a cada ensayo.
- Identificador de ensayo, fecha, configuración, hardware, firmware, condiciones de sujeción y modo real del autopiloto.
- Una nota que indique si el ensayo fue **realizado**, **simulado**, **analizado retrospectivamente**, **propuesto** o **pendiente**.

No se incorporará todo el dossier de 400 páginas al contexto activo de desarrollo. Este registro será el índice conciso de antecedentes y se ampliará únicamente cuando se identifique una evidencia concreta.


## 8. Principios de desarrollo

- Mantener el alcance acotado.
- Priorizar código claro, directo y verificable.
- Evitar capas, módulos, supervisores o abstracciones que no aporten una función demostrable.
- Separar código de pruebas.
- Priorizar pruebas offline con vídeo y datos sintéticos.
- No duplicar funcionalidades sin justificación.
- No convertir cada cambio en una auditoría completa del proyecto.
- Entregar cambios concretos, ejecutables en el ámbito de pruebas autorizado y con instrucciones claras.
- Explicar brevemente los archivos modificados y las pruebas realizadas.
- No afirmar que una prueba se ha ejecutado si no se ha ejecutado.
- No modificar archivos de referencia ni crear variantes innecesarias.
- Mantener documentación de decisiones, dependencias y estado.

## 9. Organización documental mínima

Se propone mantener pocos documentos activos:

- `README.md`: punto de entrada, resumen y estado inicial.
- `00_DOCUMENTO_MAESTRO_TFG_DRON_NUEVO.md`: este documento.
- `calibracion_runcam.npz`: calibración de cámara utilizada por los análisis visuales (ubicación documentada).
- `ESTADO_PROYECTO.md`: estado actual, decisión reciente y siguiente paso.
- `requirements.txt`/equivalente: dependencias, una vez verificadas.
- Código y pruebas offline.
- Datos de prueba y resultados, organizados sin mezclar con código fuente.

El historial anterior se conserva externamente.

## 10. Criterio de finalización de la primera etapa

La primera etapa se considerará completada cuando exista una base de software de percepción/estimación que:

- pueda ejecutarse en el entorno documentado;
- tenga pruebas offline reproducibles;
- procese datos de vídeo o conjuntos de prueba;
- muestre y registre resultados;
- documente sus limitaciones;
- no dependa de una conexión al dron real para las pruebas de esta etapa.

La integración simulada será una etapa independiente y se definirá después de consolidar esta base.

---

**Directriz para asistentes de desarrollo:** trabajar sobre la tarea concreta solicitada y priorizar la producción de un entregable. No iniciar auditorías generales, reabrir decisiones previas sin motivo concreto, ampliar el alcance de forma espontánea ni generar múltiples variantes de archivos. Cuando una cuestión de seguridad, compatibilidad o incertidumbre técnica sea relevante, señalarla de forma breve, concreta y verificable, sin convertir la respuesta en un nuevo ciclo de auditoría.
