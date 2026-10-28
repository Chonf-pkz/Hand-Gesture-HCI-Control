import pandas as pd
from pathlib import Path
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from features import extract_features

def draw_bbox(image, bboxes):
    annotated_image = image.copy()

    image_height, image_width = image.shape[:2]

    for x, y, w, h in bboxes:
        x_pixel = x * image_width
        y_pixel = y * image_height
        bbox_width = w * image_width
        bbox_height = h * image_height

        top_left = (int(x_pixel), int(y_pixel))
        bottom_right = (
            int(x_pixel + bbox_width),
            int(y_pixel + bbox_height),
        )

        cv.rectangle(annotated_image, top_left, bottom_right, (0, 255, 0), 2)
    
    return annotated_image


df = pd.read_json("data/raw/hagrid-sample-30k-384p/ann_train_val/call.json", orient="index")

folder = Path("data/raw/hagrid-sample-30k-384p/hagrid_30k/train_val_call")
image_list = list(folder.glob("*.jpg"))

lack_label = [p.stem for p in image_list
              if p.stem not in df.index]
image_ids = [p.stem for p in image_list
             if p.stem in df.index]

print("The number of image lack label: ",len(lack_label))
print("The number of image have label: ",len(image_ids))
print(df.loc[image_ids, ["labels","bboxes","user_id"]].head(3))

base_options = python.BaseOptions(
        model_asset_path="artifacts/hand_landmarker.task"
    )
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)
detector=vision.HandLandmarker.create_from_options(options)

for i in range(5):
    image_id = image_ids[i]
    image = cv.imread(f"{folder}/{image_id}.jpg")
    bboxes = df.loc[image_id, "bboxes"]

    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )
    result = detector.detect(mp_image)
    print("The id of the image: ", image_id)
    print("The number of box in the image: ", len(bboxes))
    print("The number of hand in the image: ", len(result.hand_landmarks))

    annotated_image = draw_bbox(image, bboxes)
    height, width, channels = image.shape
    
    for hand in result.hand_landmarks:
        features = extract_features(hand, width, height)
        if features is not None:
            print(len(features))
            print(features[:3])
    # print(result.hand_landmarks) 
    # This line to show the list in the result, that is present the output of the detector got which hand that is recognited
    cv.imshow("bounding box image", annotated_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

detector.close()