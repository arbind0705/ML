import cv2
import face_recognition as fr
import numpy as np

image = fr.load_image_file("arbind.jpeg")

encodings = fr.face_encodings(image)

if len(encodings) == 0:
    print("no face found in image")
    exit()

Known_face_encodings = [encodings[0]]
Known_face_names = ["Arbind"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, 
                        cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    
    rgb_frame = cv2.cvtColor(frame,
                             cv2.COLOR_BGR2RGB)
    
    for (x, y, w, h) in faces:
        
        top = y
        right = x + w
        bottom = y + h
        left = x
        
        encodings = fr.face_encodings(
            rgb_frame,
            [(top, right, bottom, left)]
        )
        
        if len(encodings) == 0:
            continue
        
        face_encoding = encodings[0]
        
        matches = fr.compare_faces(
            Known_face_encodings,
            face_encoding
        )
        name = "unknown"
        
        face_distances = fr.face_distance(
            Known_face_encodings, 
            face_encoding
        )
        
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = Known_face_names[best_match_index]
            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                    2
                )
            cv2.putText(
                frame,
                name,
                (left, top -10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        cv2.imshow("Face Recognitation ", frame)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
cap.release()
cv2.destroyALLindows()