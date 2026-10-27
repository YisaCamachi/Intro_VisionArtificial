# Intro_VisionArtificial
# Intro_VisionArtificial

Práctica de laboratorio de **Visión Artificial** del curso Matemática & Simulación Científica (Deep Learning Workshop).

**Autor:** [Tu nombre]

## Descripción

Introducción a la visión artificial con Python: captura de video desde la webcam, detección de rostros con OpenCV (Haar Cascade) y detección de objetos en tiempo real con YOLO (Ultralytics).

## Herramientas

| Herramienta | Uso |
|---|---|
| Python 3.10+ | Lenguaje principal |
| OpenCV (`opencv-python<5`) | Captura de cámara y procesamiento de imagen |
| Haar Cascade | Detección básica de rostros |
| Ultralytics YOLO (`yolov8n.pt`) | Detección de objetos |

## Instalación

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> En Linux/macOS: `python3 -m venv .venv` y `source .venv/bin/activate`.

## Scripts

| Archivo | Descripción |
|---|---|
| `01_webcam_basica.py` | Abre la webcam y muestra el video |
| `02_detector_rostros.py` | Detecta rostros con Haar Cascade |
| `03_detector_objetos_yolo.py` | Detección de objetos con YOLO en tiempo real |
| `04_yolo_guardar_video.py` | Detección con YOLO guardando el resultado en video |
| `05_contador_personas.py` | Cuenta personas en cámara |
| `06_yolo_personalizado.py` | YOLO frame por frame, con filtro por clases |
| `07_detector_celulares.py` | Cuenta celulares detectados |
| `08_rostros_desenfocados.py` | Desenfoca los rostros detectados |

Ejecutar cualquiera con `python nombre_del_archivo.py`. Para salir, presionar **Q** sobre la ventana.

## Configuración usada

- Modelo: `yolov8n.pt` (nano)
- Confianza mínima (`conf`): 0.5 (0.45 en el detector de celulares)
- Haar Cascade: `scaleFactor=1.1`, `minNeighbors=5`, `minSize=(60, 60)`

## Limitaciones

- Haar Cascade falla con poca luz, rostros de lado, mascarillas o gafas, y da falsos positivos.
- YOLO nano es rápido pero menos preciso que modelos más grandes.
- Estos scripts son educativos y no deben usarse como sistema de seguridad real.

## Uso ético

Este proyecto **detecta** rostros y objetos, no **reconoce identidades**. No se debe grabar ni publicar rostros sin autorización. Los videos generados (`runs/`) están excluidos del repositorio mediante `.gitignore`.
