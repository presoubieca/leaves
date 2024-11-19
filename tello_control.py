from ultralytics import YOLO
import torch
import cv2
import time
from djitellopy import Tello
from PIL import Image
import os

dir = os.getcwd()
model_dir = dir + r"\runs\detect\train_high_Res\weights\best.pt"
model = YOLO(model_dir, task="detect")

tello = Tello()
tello.connect()
tello.streamon()
tello.get_frame_read()
time.sleep(0.5)
i = 0
print(str(tello.get_battery()))
rot1,rot2,rot3 = 0,0,0
tello.takeoff()

while i <= 20:

#    if tello.get_height() < 100:
#        print(str(100 - tello.get_height()))
#        tello.move_up(int(100-tello.get_height()))

    if i >= 5 and rot1 == 0:
        tello.rotate_clockwise(90)
        time.sleep(1)
        i = i + 1
        rot1 = 1
    if i >= 10 and rot2 == 0 and rot1 == 1:
        tello.rotate_clockwise(90)
        time.sleep(1)
        i = i + 1
        rot2 = 1
    if i >= 15 and rot3 == 0 and rot1 == 1 and rot2 == 1:
        tello.rotate_clockwise(90)
        time.sleep(1)
        i = i + 1
        rot3 = 1

    time.sleep(0.5)
    i = i + 0.5
    image = tello.get_frame_read()
    cv2.imwrite("test/picture"+str(i)+".jpg", image.frame)
    time.sleep(1)
    i = i + 1
    print(i)

tello.land()
tello.streamoff()
for img in os.listdir(dir+r"/test"):
    if img.endswith(".jpg") or img.endswith(".png"):
        path = dir+r"\\test\\" + img

        results = model.predict(path, save=False, conf=0.4)

        for result in results:
            i = 0
            image = Image.open(path)
            cords = torch.round(result.boxes.xyxy).to(torch.int)
            for x in cords:
                i += 1
                box = image.crop(x.tolist())

                if box.size[0] >= 80 and box.size[1] >= 80:
                    if not box.size[0]*2.5 < box.size[1] and not box.size[1]*2.5 < box.size[0]:
                        box.save("test_result\\"+img[:]+"___"+str(i)+".jpg", verbose=0)

    else:
        print("Wrong format")
