from ultralytics import YOLO
import cv2

modelo = YOLO("yolov8n.pt")
camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = camara.read()

    if not ret:
        break

    resultados = modelo(frame, conf=0.45, verbose=False)

    celulares = 0

    for caja in resultados[0].boxes:
        clase_id = int(caja.cls[0])
        nombre = modelo.names[clase_id]

        if nombre == "cell phone":
            celulares += 1

    frame_anotado = resultados[0].plot()

    cv2.putText(
        frame_anotado,
        f"Celulares detectados: {celulares}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    cv2.imshow("Detector de celulares - Q para salir", frame_anotado)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()