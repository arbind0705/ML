import cv2
import face_recognition as fr

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    rgb = frame[:, :, ::-1]

    locations = fr.face_locations(rgb)

    print("Locations:", locations)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()