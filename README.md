Este repositorio inicia una nueva etapa de desarrollo del TFG. Su referencia de partida es el archivo `AUD-05_v5.22_vision_externa_candidata.py`, conservado como copia de referencia.
Objetivo
Estudiar una arquitectura de percepción visual basada en cámara FPV y marcador ArUco, procesamiento en Python, validación de observaciones y estimación temporal, con una futura integración en un entorno de simulación representativo.
El objetivo inmediato es construir y comprobar componentes de software de forma reproducible, sin repetir auditorías generales ni reconstruir el historial de los siete chats anteriores.
Arquitectura conceptual
Adquisición de imagen de cámara FPV.
Detección ArUco.
Validación geométrica, dimensional y temporal.
Estimación y filtrado de la observación.
Gestión de estados de software y calidad de percepción.
Registro de datos y pruebas offline.
Etapa posterior: integración y experimentación en simulación.
El esquema es conceptual y no constituye una configuración de vuelo.
Hardware y software de referencia
Cuadricóptero F450.
Pixhawk / ArduCopter (versiones y configuración concreta se documentarán cuando se verifiquen).
Cámara RunCam FPV y enlace analógico de vídeo de 5,8 GHz.
Mando RC.
Python, OpenCV/ArUco y MAVLink/PyMAVLink como tecnologías del proyecto.
VS Code, Git y GitHub.
Ubuntu/WSL y ArduPilot SITL como herramientas de desarrollo y pruebas, sujeto a la configuración reproducible que se establezca.
Simulación (fase posterior)
La simulación se abordará después de consolidar la base de percepción y pruebas offline. Se buscará un entorno que permita visualizar un vehículo, un escenario interior y un marcador, y estudiar la interacción con un autopiloto simulado y el mando. FlightGear, Gazebo y otras alternativas no se consideran decisiones cerradas; se seleccionará una plataforma tras valorar representatividad, integración, mantenimiento y reproducibilidad.
Estado
[x] Crear repositorio independiente.
[ ] Incorporar el documento maestro (`00_DOCUMENTO_MAESTRO_TFG_DRON_NUEVO.md`).
[ ] Incorporar la copia de referencia `AUD-05_v5.22_vision_externa_candidata.py`.
[ ] Incorporar `calibracion_runcam.npz` y definir/documentar su ruta.
[ ] Organizar las evidencias anteriores que se decida conservar (logs, gráficos, vídeos y `.BIN`) por ensayo.
[ ] Establecer entorno reproducible y pruebas offline.
[ ] Consolidar visión/estimación y registro de datos.
[ ] Diseñar y desarrollar la simulación en una etapa independiente.
Datos y antecedentes técnicos
`calibracion_runcam.npz`: archivo de calibración de la cámara RunCam (parámetros intrínsecos y de distorsión, si se confirma el contenido del archivo). Debe mantenerse en la ruta acordada por el proyecto y no reemplazarse silenciosamente por valores estimados.
El documento maestro registra un resumen de los scripts y pruebas anteriores, incluyendo la comparación de controladores, ejemplos de Kalman, pruebas SITL, ensayos de banco y pruebas de visión/buffering. Se diferencia entre ejecución documentada, simulación, análisis retrospectivo y trabajo pendiente.
Los archivos de evidencia (logs, vídeos, gráficas y `.BIN`) se conservarán organizados por ensayo; no se copiarán indiscriminadamente al repositorio principal.
Documentación
`00_DOCUMENTO_MAESTRO_TFG_DRON_NUEVO.md`: objetivo, contexto, arquitectura conceptual, componentes, criterios de desarrollo y alcance.
`README.md`: resumen y punto de entrada del repositorio.
`calibracion_runcam.npz`: archivo de calibración óptica, que se utilizará únicamente con su procedencia y estado documentados.
Criterio de rigor
Se distinguirá entre decisiones de diseño, valores propuestos, resultados de pruebas offline, resultados de simulación, pruebas de banco y experimentación física. Ningún valor o resultado se considerará validado sin evidencia documentada.
