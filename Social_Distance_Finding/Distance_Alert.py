import torch
import torchvision
import os
import sys
import math
import argparse
import time
import cv2 as cv
import numpy as np

from pathlib import Path
from torchvision import transforms

from yolo_v3.yolov3_model import YOLOv3
from yolo_v3.utils import calculate_iou, nms, transform_predicted_txtytwth


parser = argparse.ArgumentParser(description="This is for the arguments required for the inference of yolov3.")
# parser.add_argument("-t", "--path", type=str, default="data/video.mp4", help="Argument for the path to the input frames.")
parser.add_argument("-n", "--nms_conf", type=float, default=0.5, help="Argument to provide confidence threshold to perform nms")
parser.add_argument("-p", "--plot_conf", type=float, default=0.70, help="Argument to plot the predictions which have given confidence.")
parser.add_argument("-i", "--input_size", type=int, default=608, help="Argument for the size of input image to model.")
parser.add_argument("-w", "--wts", type=str, default="yolov3_wts_reference.pt", help="Argument for the path to trained weights.")

arg = parser.parse_args()

CLASSES = []
with open("./coco.names", "r") as fp:
    for line in fp.readlines():
        CLASSES.append(line.strip('\n'))
# print(f'Classes are : \n {CLASSES}')        

def inference(model, image_ori, transform, size):
    image = cv.resize(image_ori, (arg.input_size, arg.input_size))
    transformed_image = transform(image)
    input_image = transformed_image.unsqueeze(dim=0)
    model.eval()
    with torch.inference_mode():
        out = model(input_image)
    device = out[0].device
    l = out[0]
    m = out[1]
    s = out[2]
    pred_bboxes_large = transform_predicted_txtytwth(bboxes=l, grid_size=size, device=device, scale="large")
    pred_bboxes_large = pred_bboxes_large.flatten(0,3)
    pred_bboxes_large = pred_bboxes_large.tolist()
    filtered_bboxes_large = nms(pred_bboxes=pred_bboxes_large, prob_threshold=arg.nms_conf, iou_threshold=0.3, format="center")

    pred_bboxes_medium = transform_predicted_txtytwth(bboxes=m, grid_size=size, device=device, scale="medium")
    pred_bboxes_medium = pred_bboxes_medium.flatten(0,3)
    pred_bboxes_medium = pred_bboxes_medium.tolist()
    filtered_bboxes_medium = nms(pred_bboxes=pred_bboxes_medium, prob_threshold=arg.nms_conf, iou_threshold=0.3, format="center")

    pred_bboxes_small = transform_predicted_txtytwth(bboxes=s, grid_size=size, device=device, scale="small")
    pred_bboxes_small = pred_bboxes_small.flatten(0,3)
    pred_bboxes_small = pred_bboxes_small.tolist()
    filtered_bboxes_small = nms(pred_bboxes=pred_bboxes_small, prob_threshold=arg.nms_conf, iou_threshold=0.3, format="center")

    all_predictions = [filtered_bboxes_large, filtered_bboxes_medium, filtered_bboxes_small]
    final_boxes = []
    for idx, scale_box in enumerate(all_predictions):
        for box in scale_box:
            confidence = box[4]
            x_c = (box[0]/size[idx])
            y_c = (box[1]/size[idx])
            w = (box[2]/608)
            h = (box[3]/608)
            class_prob = box[5:]
            label = torch.argmax(torch.softmax(torch.tensor(class_prob), dim=-1),dim=-1)

            x_min = (x_c - (w/2))*image_ori.shape[1]
            y_min = (y_c - (h/2))*image_ori.shape[0]
            x_max = (x_c + (w/2))*image_ori.shape[1]
            y_max = (y_c + (h/2))*image_ori.shape[0]
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = max(0, x_max)
            y_max = max(0, y_max)
            
            if confidence > arg.plot_conf:
                final_boxes.append([x_min, y_min, x_max, y_max, label.item()])
    
    return final_boxes

def calculate_dist(pt1, pt2):
    diff1 = (pt2[0]-pt1[0])**2
    diff2 = (pt2[1]-pt1[1])**2
    dist = diff1 + diff2
    distance = np.pow(dist,0.5)

    return distance.item()

def find_distance(image, bboxes, threshold_dist): 
    red_set = set()
    for i in range(len(bboxes)):
        box = bboxes[i]
        x_c = (box[2]+box[0])/2
        y_c = (box[3]+box[1])/2
        for j in range(len(bboxes)):
            if i == j:
                continue
            bx = bboxes[j]
            x_center = (bx[2]+bx[0])/2
            y_center = (bx[3]+bx[1])/2
            dst = calculate_dist((x_c, y_c), (x_center, y_center))
            # print(f"index and distance: {(i, j)}__{dst}")
            if dst < threshold_dist:
                color = (0,0,255)
                cv.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 3)
                cv.rectangle(image, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), color, 3)
                cv.line(image, (int(x_c), int(y_c)), (int(x_center), int(y_center)), color, 3)
                red_set.add(j)
                red_set.add(i)
                # print(f'red index: {red_set}')
            if dst >= threshold_dist:
                color = (0,255,0)
                if red_set == set():
                    cv.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 3)
                    cv.rectangle(image, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), color, 3)
                    cv.line(image, (int(x_c), int(y_c)), (int(x_center), int(y_center)), color, 3)
                else:
                    if i not in red_set and j not in red_set:
                        cv.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 3)
                        cv.rectangle(image, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), color, 3)
                        cv.line(image, (int(x_c), int(y_c)), (int(x_center), int(y_center)), color, 3)
                    elif i in red_set and j not in red_set:
                        cv.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 3)
                        cv.rectangle(image, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), color, 3)
                        cv.line(image, (int(x_c), int(y_c)), (int(x_center), int(y_center)), color, 3)
                    elif i not in red_set and j in red_set:
                        cv.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 3)
                        cv.rectangle(image, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0,0,255), 3)
                        cv.line(image, (int(x_c), int(y_c)), (int(x_center), int(y_center)), color, 3)
                    elif i in red_set and j in red_set:
                        cv.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 3)
                        cv.rectangle(image, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0,0,255), 3)
                        cv.line(image, (int(x_c), int(y_c)), (int(x_center), int(y_center)), color, 3)
        
                        
            # cv.imshow("visualize", image)
            # cv.waitKey(0)
       
    return image        
    # cv.imwrite("./distance_visualize2.jpg", image)
    # cv.waitKey(0)      
                    
            
def convert(box):
    xmin = box[0] - box[2]/2
    ymin = box[1] - box[3]/2  
    xmax = box[0] + box[2]/2
    ymax = box[1] + box[3]/2   
    return [xmin, ymin, xmax, ymax]
      
    
def main():
    anchors_list = [
            [116,90, 156,198, 373,326],
            [30,61, 62,45, 59,119],
            [10,13, 16,30, 33,23]]
    grid_size = [arg.input_size//32, arg.input_size//16, arg.input_size//8]
    num_class = 80
    weight_path = arg.wts

    transformation = transforms.Compose([
        transforms.ToTensor()
    ])

    wts = torch.load(f=weight_path, map_location=torch.device('cpu'), weights_only=True)
    model = YOLOv3(input_size=arg.input_size, anchors=anchors_list, num_classes=num_class)
    # print(len(model.state_dict().keys()))
    new_wts = {my_para_name:wts[pretrained_para_name] for my_para_name, pretrained_para_name in zip(model.state_dict().keys(), wts.keys())}
    model.load_state_dict(state_dict=new_wts, strict=True)
    # print(f'[INFO] The pretrained weight are loaded successfully from path {weight_path}.')
    
    
    image = cv.imread("data/example3.jpg", cv.IMREAD_COLOR)
    bboxes = inference(model=model, image_ori=image, transform=transformation, size=grid_size)
    assert type(bboxes) == list , "Invalid type returned from inference function it must be List[list]."
    final_img = find_distance(image=image, bboxes=bboxes, threshold_dist=200)
    cv.imwrite("test_on_rgb_image2.jpg", final_img)
    
    
    # video_file = arg.path
    # cap = cv.VideoCapture(video_file)
    # if not cap.isOpened():
    #     print(f'cannot open the video')
    #     exit()
        
    # cv.namedWindow('video', cv.WINDOW_NORMAL)
    # cv.resizeWindow('video', 608,608)
    # threshold_dist = 200
    # while True:
    #     # image = cv.imread("data/pedestrain1.jpg", cv.IMREAD_COLOR)
    #     ret, image = cap.read()
    #     if not ret:
    #         print('video ended.')
    #         exit()
            
    #     bboxes = inference(model=model, image_ori=image, transform=transformation, size=grid_size)
    #     assert type(bboxes) == list , "Invalid type returned from inference function it must be List[list]."
    #     # print(f'bboxes from inference in list[list] format: {bboxes}')
    #     final_img = find_distance(image=image, bboxes=bboxes, threshold_dist=threshold_dist)
    #     cv.imshow("video", final_img)
        
    #     key = cv.waitKey(1)&0xff 
    #     if key == ord('a'):
    #         threshold_dist += 50
    #     elif key == ord('s'):
    #         threshold_dist -= 50
    #     elif key == ord('q'):
    #         break
        
    
    # cap.release()
    # cv.destroyAllWindows()

    # dummy_image = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    # boxes = [[250, 150, 100, 100], [600, 150, 100, 100], [250, 450, 100, 100], [1050, 650, 100, 100], [450,400,100,100]] # 300=threshold
    # boxes = [[250, 150, 100, 100], [400, 150, 100, 100], [250, 380, 100, 100], [1050, 650, 100, 100], [850,400,100,100]] #350=threhold
    # boxes = [[250, 150, 100, 100], [1100, 150, 100, 100], [250, 380, 100, 100], [1050, 650, 100, 100], [850,400,100,100]] #400=threhold
    
    # dummy_bboxes = []
    # for box in boxes:
    #     dummy_bboxes.append(convert(box=box))
    # find_distance(image=dummy_image, bboxes=dummy_bboxes, threshold_dist=300)
    # for (x_min, y_min, x_max, y_max) in dummy_bboxes:
    #     cv.rectangle(dummy_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

    # output_path = "dummy_image_with_bboxes.png"
    # cv.imwrite(output_path, dummy_image)
    

    
if __name__ == "__main__":
    main()