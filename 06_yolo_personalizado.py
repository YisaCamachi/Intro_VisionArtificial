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

    resultados = modelo(frame, conf=0.5, verbose=False, classes=[63])
#0 personas, 24 backpacks, 39 botellas, 41 copas, 63 laptops, 67 celulares, 70 silla, 73 libros, 72 sofás, 73 mesas, 74 televisores
    frame_anotado = resultados[0].plot()

    cv2.imshow("YOLO personalizado - presiona Q para salir", frame_anotado)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()