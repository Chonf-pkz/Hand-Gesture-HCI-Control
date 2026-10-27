import cv2 as cv

camera = cv.VideoCapture(0)

while True:
    success, frame = camera.read()
    if success:
        cv.imshow("camera", frame)
    else:
        print("Can't connect to the camera")
        break
    if cv.waitKey(1) == 27:
        break

camera.release()
cv.destroyAllWindows()