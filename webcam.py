import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

base_options = python.BaseOptions(
    model_asset_path="artifacts/hand_landmarker.task",
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)
camera = cv.VideoCapture(0)
last_timestamp_ms = -1
connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

while True:
    success, frame = camera.read()
    if success:
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )
        timestamp_ms = max(
            last_timestamp_ms + 1,
            time.monotonic_ns() // 1_000_000
        )
        last_timestamp_ms = timestamp_ms
        result = detector.detect_for_video(mp_image, timestamp_ms)
        # print(result.hand_landmarks) 
        # This line to show the list in the result, that is present the output of the detector got which hand that is recognited
        
        height, width, channels = frame.shape
        for hand in result.hand_landmarks:
            for point in hand:
                pixel_x = int(point.x * width)
                pixel_y = int(point.y * height)
                cv.circle(frame, (pixel_x,pixel_y), 4, (0,255,0),-1)

            for connection in connections:
                start_point = hand[connection.start]
                end_point = hand[connection.end]
                start_pixel = (
                    int(start_point.x * width),
                    int(start_point.y * height)
                )
                end_pixel = (
                    int(end_point.x * width),
                    int(end_point.y * height)
                )
                cv.line(frame, start_pixel, end_pixel, (255,255,255), 2)

        cv.imshow("camera", frame)
    else:
        print("Can't connect to the camera")
        break
    if cv.waitKey(1) == 27:
        break

detector.close()
camera.release()
cv.destroyAllWindows()