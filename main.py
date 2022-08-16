import pathlib
# cv imports for motion detection
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# pytorch imports for monkey detection
from torchvision import models, transforms
import torch
from PIL import Image
# GPIO motor imports for linear actuator
from gpiozero import Motor
# time function
import time

def check_for_monkey(img):
    dir(models)
    alexnet = models.alexnet(pretrained=True)

    transform = transforms.Compose([                #[1]
        transforms.Resize(256),                     #[2]
        transforms.CenterCrop(224),                 #[3]
        transforms.ToTensor(),                      #[4]
        transforms.Normalize(                       #[5]
        mean=[0.485, 0.456, 0.406],                 #[6]
        std=[0.229, 0.224, 0.225]                   #[7]
    )])

    color_coverted = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)
    img_t = transform(pil_image)
    batch_t = torch.unsqueeze(img_t, 0)

    alexnet.eval()
    out = alexnet(batch_t)

    # Load labels
    with open(str(pathlib.Path().resolve()) + '/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
        _, indices = torch.sort(out, descending=True)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

    prediction = [(classes[idx], percentage[idx].item()) for idx in indices[0]]
    # print("Prediction ", prediction)

    for p in prediction:
        pid = 0
        if p[0].split(',')[0].isdigit():
            pid = int(p[0].split(',')[0])
        score = p[1]
        if 365 <= pid <= 382 and score > 1.5:
            print('monkey detected')
            return True

    print('no monkey detected')
    return False

def close_window():
    motor1 = Motor(forward=4, backward=14)
    motor1.forward()
    time.sleep(10)

def motion_detection():
    cap = cv.VideoCapture(0)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    last_check_time = 0

    while cap.isOpened():
        diff = cv.absdiff(frame1, frame2)
        diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(diff_gray, (5, 5), 0)
        _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
        dilated = cv.dilate(thresh, None, iterations=3)
        contours, _ = cv.findContours(
            dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        is_motion = False
        for contour in contours:
            (x, y, w, h) = cv.boundingRect(contour)
            if cv.contourArea(contour) < 900:
                continue
            cv.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 3)
            is_motion = True

        cv.imshow("Video", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()

        if is_motion and time.time() >= last_check_time + 5:
            last_check_time = time.time()
            print(time.ctime(last_check_time))
            if check_for_monkey(frame2) is True:
                close_window()
                break

        if cv.waitKey(50) == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    motion_detection()