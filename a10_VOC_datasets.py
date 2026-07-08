from torchvision.datasets import VOCDetection
from pprint import pprint
from torchvision.transforms import Compose, ToTensor
import torch

class VOCDataset(VOCDetection):
    def __init__(self,root, year, image_set, download, transform):
        super().__init__(root, year, image_set, download,transform)
        self.categories = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
                      "chair","cow","diningtable","dog","horse","motorbike","person","pottedplant",
                      "sheep","sofa","train","tvmonitor"]

    def __getitem__(self, item):
        image, data = super().__getitem__(item)
        all_bbox = []
        all_label = []
        for obj in data["annotation"]["object"]:
            xmin = int(obj["bndbox"]["xmin"])
            ymin = int(obj["bndbox"]["ymin"])
            xmax = int(obj["bndbox"]["xmax"])
            ymax = int(obj["bndbox"]["ymax"])
            all_bbox.append([xmin,ymin,xmax,ymax])
            all_label.append(self.categories.index(obj["name"]))

        all_bbox = torch.FloatTensor(all_bbox)
        all_label = torch.LongTensor(all_label)
        target = {
            "boxes": all_bbox,
            "labels": all_label
        }
        return image, target

if __name__ == '__main__':
    transform = ToTensor()
    dataset = VOCDataset(root=r"C:\Users\Admin\Desktop\PythonProject\data\VOC", year="2012",image_set="train",
                         download=False, transform = transform)
    image, label = dataset[2000]
    pprint(label)
    print(image.shape)
