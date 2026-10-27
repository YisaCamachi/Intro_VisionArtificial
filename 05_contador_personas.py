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

    resultados = modelo(frame, conf=0.5, verbose=False)

    conteo_personas = 0

    for caja in resultados[0].boxes:
        clase_id = int(caja.cls[0])
        nombre = modelo.names[clase_id]

        if nombre == "person":
            conteo_personas += 1

    frame_anotado = resultados[0].plot()

    cv2.putText(
        frame_anotado,
        f"Personas: {conteo_personas}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Contador de personas - Q para salir", frame_anotado)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()