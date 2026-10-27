import cv2

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

ruta_cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detector_rostros = cv2.CascadeClassifier(ruta_cascade)

if detector_rostros.empty():
    print("No se pudo cargar el clasificador Haar Cascade.")
    exit()

while True:
    ret, frame = camara.read()

    if not ret:
        print("No se pudo leer el frame.")
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rostros = detector_rostros.detectMultiScale(
        gris,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in rostros:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Rostro",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Detector de rostros - presiona Q para salir", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()