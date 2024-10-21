import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./Model/Model.yml')
cascadePath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

cam = cv2.VideoCapture(0)

with open('Name_list/Name.txt', 'r') as f:
    data = f.read()

names = data.split(',')

names = [name.strip() for name in names if name.strip()]

print(names)

while True:
    
    ret,frame = cam.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:


        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray_frame[y:y+h, x:x+w])

        if confidence < 100:
            id = names[id]
            confidence = "{0}%".format(round(confidence))

        else:
            id = "unknown"
            confidence = "{0}%".format(round(confidence))

        cv2.putText(frame, str(id), (x-5,y-5), font, 1, (255,0,255), 2)
        cv2.putText(frame, str(confidence), (x+150,y-5), font, 1, (255,0,255), 2)

    cv2.imshow("Face detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

