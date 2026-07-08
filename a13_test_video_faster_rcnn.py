import os
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from argparse import ArgumentParser
import cv2

categories = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
                      "chair","cow","diningtable","dog","horse","motorbike","person","pottedplant",
                      "sheep","sofa","train","tvmonitor"]
def get_args():
    parser = ArgumentParser(description="train FASTER RCNN")
    parser.add_argument("--train_model", "-l", type = str, default="train_model/best_faster_rcnn.pt")
    parser.add_argument("--conf_threshold", "-c", type = float, default=0.7)
    args = parser.parse_args()
    return args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test():

    args = get_args()
    model = fasterrcnn_mobilenet_v3_large_320_fpn()
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes=21)
    model.to(device)

    if args.train_model:
        checkpoint = torch.load(args.train_model)
        model.load_state_dict(checkpoint["model"])
        print("load_success")
    else:
        print("load_false")

    cap = cv2.VideoCapture(r"C:\Users\Admin\Videos\test_1\test_1-1.mp4")
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter("result_1.mp4", cv2.VideoWriter_fourcc(*"mp4v"), int(cap.get(cv2.CAP_PROP_FPS)),
                          (width,height))
    model.eval()
    while cap.isOpened():
        flat, frame = cap.read()
        if not flat:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2,0,1))
        image = [torch.from_numpy(image).to(device).float()/255]    # input_model : list (tensor)

        with torch.no_grad():
            output = model(image)[0]
            boxes = output["boxes"]
            labels = output["labels"]
            scores = output["scores"]
            for box, label, score in zip(boxes, labels, scores):
                if score > args.conf_threshold:
                    x_min, y_min,x_max, y_max = box
                    cv2.rectangle(frame,
                                  (int(x_min), int(y_min)),
                                  (int(x_max), int(y_max)),
                                  (0, 0, 255),
                                  3)
                    category = categories[label]
                    text = f"{category}: {score:.2f}"
                    cv2.putText(frame,text,(int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)
        out.write(frame)

    cap.release()
    out.release()

#19:04

if __name__ == '__main__':
    test()