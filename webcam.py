import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from features import extract_features
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from features import extract_features

def calculate_iou(box_a, box_b):
    S_a = box_a[2] * box_a[3]
    S_b = box_b[2] * box_b[3]
    left = max(box_a[0], box_b[0])
    top = max(box_b[0], box_b[1])
    right = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    bottom = min(box_b[0] + box_a[3], box_b[1] + box_b[3])
    S_i = max(0, right - left) * max(0, bottom - top)

    return (S_i / (S_a + S_b - S_i))

def get_hand_bboxes(hands):
    bboxes = []

    if not hands:
        return bboxes
    else:
        for hand in hands:
            xs = [point.x for point in hand]
            ys = [point.y for point in hand]

            x = min(xs)
            y = min(ys)
            w = max(xs) - x
            h = max(ys) - y

            bboxes.append([x,y,w,h])

    if len(bboxes) > 1:
        box_a = bboxes[0]
        box_b = bboxes[1]
        iou = calculate_iou(box_a, box_b)
    
        print(f"IoU: {iou}")

    return bboxes

def draw_bbox(frame, bboxes):
    frame_height, frame_width = frame.shape[:2]

    for x, y, w, h in bboxes:
        x_pixel = x * frame_width
        y_pixel = y * frame_height
        bbox_width = w * frame_width
        bbox_height = h * frame_height

        top_left = (int(x_pixel), int(y_pixel))
        bottom_right = (
            int(x_pixel + bbox_width),
            int(y_pixel + bbox_height),
        )

        cv.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    
    return frame

def draw_hands(frame, hands, connections):
    annotated_frame = frame.copy()

    height, width, channels = annotated_frame.shape
    for hand in hands:
        for point in hand:
            pixel_x = int(point.x * width)
            pixel_y = int(point.y * height)
            cv.circle(annotated_frame, (pixel_x,pixel_y), 4, (0,255,0),-1)

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
            cv.line(annotated_frame, start_pixel, end_pixel, (255,255,255), 2)

    values = get_hand_bboxes(hands)

    return annotated_frame, values

def main():
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
            height, width, channels = frame.shape

            # for hand in result.hand_landmarks:
            #     features = extract_features(hand, width, height)
            #     if features is not None:
            #         print(len(features))
            #         print(features[:3])
            # print(result.hand_landmarks) 
            # This line to show the list in the result, that is present the output of the detector got which hand that is recognited
            annotated_frame, values = draw_hands(
                frame,
                result.hand_landmarks,
                connections
            )
            annotated_frame = draw_bbox(annotated_frame, values)
            cv.imshow("camera", annotated_frame)
        else:
            print("Can't connect to the camera")
            break
        if cv.waitKey(1) == 27:
            break

    detector.close()
    camera.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()