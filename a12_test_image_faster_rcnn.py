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
    parser.add_argument("--conf_threshold", "-c", type = float, default=0.5)
    args = parser.parse_args()
    return args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test():

    args = get_args()
    org_image = cv2.imread(r"C:\Users\Admin\Desktop\BO_SUU_TAP\z7137343290849_712465394bc9ec2ff185f1130595c522.jpg") #BRG
    image = org_image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2,0,1))
    image = [torch.from_numpy(image).to(device).float()/255]    # input_model : list (tensor)

    model = fasterrcnn_mobilenet_v3_large_320_fpn()
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes= 21)
    model.to(device)

    if args.train_model:
        checkpoint = torch.load(args.train_model)
        model.load_state_dict(checkpoint["model"])
        print("load_success")
    else:
        print("load_false")

    model.eval()
    with torch.no_grad():
        output = model(image)[0]
        boxes = output["boxes"]
        labels = output["labels"]
        scores = output["scores"]
        for boxes, labels, scores in zip(boxes, labels, scores):
            if scores > args.conf_threshold:
                print(boxes, labels)
                x_min, y_min,x_max, y_max = boxes
                cv2.rectangle(org_image,
                              (int(x_min), int(y_min)),
                              (int(x_max), int(y_max)),
                              (0, 0, 255),
                              3)
                category = categories[labels]
                text = f"{category}: {scores:.2f}"
                cv2.putText(org_image,text,(int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
    cv2.imshow("image", org_image)
    cv2.waitKey(0)
if __name__ == '__main__':
    test()