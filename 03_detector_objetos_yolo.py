from ultralytics import YOLO

# Modelo nano: liviano y rápido, ideal para empezar.
# La primera vez se descarga automáticamente.
modelo = YOLO("yolov8s.pt")

# source=0 usa la webcam principal.
# show=True muestra la ventana con las detecciones.
# conf=0.5 ignora detecciones con confianza menor al 50%.
modelo.predict(source=0, show=True, conf=0.5)