import cv2

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

ruta_cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(ruta_cascade)

while True:
    ret, frame = camara.read()

    if not ret:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rostros = detector.detectMultiScale(
        gris,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in rostros:
        rostro = frame[y:y+h, x:x+w]
        rostro_blur = cv2.GaussianBlur(rostro, (99, 99), 30)
        frame[y:y+h, x:x+w] = rostro_blur

    cv2.imshow("Rostros desenfocados - Q para salir", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()