import os
import numpy as np
import torch

from a10_VOC_datasets import VOCDataset
from torchvision.transforms import ToTensor, Compose, Normalize, RandomAffine, ColorJitter
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import shutil
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint


def get_args():
    parser = ArgumentParser(description="train FASTER RCNN")
    parser.add_argument("--data", "-d", type = str, default="/data/VOC")
    parser.add_argument("--batch_size", "-b", type = int, default=8)
    parser.add_argument("--epochs", "-e", type = int, default=50)
    parser.add_argument("--logging", '-l', type = str, default="tensorboard")
    parser.add_argument("--train_model", "-t", type = str, default="train_model")
    parser.add_argument("--checkpoint", "-c", type = str, default= None)
    args = parser.parse_args()
    return args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def collate_fn(batch):
    image, label = zip(*batch)
    return list(image), list(label)




def train():
    args = get_args()
    epochs = args.epochs
    batch_size = args.batch_size
    train_transforms = Compose([
        ToTensor(),
    ])
    test_transforms = Compose([
        ToTensor(),
    ])
    train_dataset = VOCDataset(root="data/VOC", year="2012",image_set="train",
                               download=False, transform=train_transforms)
    val_dataset = VOCDataset(root="data/VOC", year="2012",image_set="val",
                             download=False, transform=test_transforms)

    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn = collate_fn
    )
    test_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False ,
        drop_last=False,
        collate_fn=collate_fn
    )

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)

    if not os.path.isdir(args.train_model):
        os.mkdir(args.train_model)

    witter = SummaryWriter(args.logging)


    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
                                              trainable_backbone_layers = 0)
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = len(train_dataset.categories)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes= num_classes)
    # print(model.transform)
    model.to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_map = checkpoint["map"]
    else:
        start_epoch = 0
        best_map = -1

    for epoch in range(start_epoch, epochs):
        #TRAIN
        progress_bar = tqdm(train_dataloader, colour='yellow')
        model.train()
        max_iters = len(train_dataloader)
        train_loss = []
        for iter, (images, labels) in enumerate(progress_bar):
            images = [img.to(device) for img in images]
            labels = [
                {k: v.to(device) for k, v in target.items()}
                for target in labels
            ]
            # #forward
            losses = model(images, labels)
            final_loss = sum([loss for loss in losses.values()])
            train_loss.append(final_loss.item())
            mean_loss = np.mean(train_loss)

            progress_bar.set_description(f'epoch:{epoch+1}/{epochs}____final_loss: {mean_loss:.3f}')
            witter.add_scalar(tag="train/loss", scalar_value = mean_loss, global_step = epoch * max_iters + iter )
            # print(final_loss.item())

            #backward
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()


        #VALIDATE
        model.eval()
        metric = MeanAveragePrecision(iou_type="bbox").to("cpu")
        for images, labels in test_dataloader:
            images = [img.to(device) for img in images]
            with torch.no_grad():
                outputs = model(images)
                # print(outputs)
            preds = []
            for output in outputs:
                preds.append({
                    "boxes": output["boxes"].to("cpu"),
                    "scores": output["scores"].to("cpu"),
                    "labels": output["labels"].to("cpu")
                })
            targets = []
            for label in labels:
                targets.append({
                    "boxes": label["boxes"],
                    "labels": label["labels"]
                })

            metric.update(preds, targets)
        result = metric.compute()
        pprint(result)
        witter.add_scalar("val/mAP", result["map"], epoch)
        witter.add_scalar("val/mAP_50", result["map_50"], epoch)
        witter.add_scalar("val/mAP_75", result["map_75"], epoch)

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch + 1,
            "optimizer": optimizer.state_dict(),
            "map": result["map"].item()
        }
        torch.save(checkpoint, os.path.join(args.train_model, "last_faster_rcnn.pt"))
        if result["map"] > best_map:
            best_map = result["map"]
            torch.save(checkpoint, os.path.join(args.train_model, "best_faster_rcnn.pt") )

if __name__ == '__main__':
    train()