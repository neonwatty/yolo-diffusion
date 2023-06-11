import requests
from ultralytics import YOLO
from torchvision.io import read_image
import torchvision.transforms as transforms
import cv2
from tqdm.notebook import tqdm 
import numpy as np
import matplotlib.pyplot as plt
import os
import random
parent_dir = os.getcwd()

# define transforms for processing tensors
resizer = transforms.Resize((448, 640),antialias=True)
clipper = transforms.Lambda(lambda x: x[:3])

# yolo segmentation models
available_models = {
    'large' :{'client_name':'YOLOv8l-seg.pt', 'server_name':'yolov8l-seg.pt'},
    'medium':{'client_name':'YOLOv8m-seg.pt', 'server_name':'yolov8m-seg.pt'},
    'small' :{'client_name':'YOLOv8s-seg.pt', 'server_name':'yolov8s-seg.pt'},
    'nano'  :{'client_name':'YOLOv8n-seg.pt', 'server_name':'yolov8n-seg.pt'}
}

# pull a desired yolo model
def pull_yolo_model(model_selection=available_models['large']):
    # create server url
    server_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/" + model_selection["server_name"]

    # create local save location
    local_save = "/content/" + model_selection["client_name"]

    # pull yolo model to this macine
    response = requests.get(server_url, stream=True)
    response.raise_for_status()
    with open(local_save, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

    # return local_save location to pass onto segmenter
    return local_save


# pull default yolo model
yolo_model_location = pull_yolo_model()

# available yolo models - from coco dataset
label_lookup_dict = {
  "person": 0,
  "bicycle": 1,
  "car": 2,
  "motorcycle": 3,
  "airplane": 4,
  "bus": 5,
  "train": 6,
  "truck": 7,
  "boat": 8,
  "traffic light": 9,
  "fire hydrant": 10,
  "stop sign": 11,
  "parking meter": 12,
  "bench": 13,
  "bird": 14,
  "cat": 15,
  "dog": 16,
  "horse": 17,
  "sheep": 18,
  "cow": 19,
  "elephant": 20,
  "bear": 21,
  "zebra": 22,
  "giraffe": 23,
  "backpack": 24,
  "umbrella": 25,
  "handbag": 26,
  "tie": 27,
  "suitcase": 28,
  "frisbee": 29,
  "skis": 30,
  "snowboard": 31,
  "sports ball": 32,
  "kite": 33,
  "baseball bat": 34,
  "baseball glove": 35,
  "skateboard": 36,
  "surfboard": 37,
  "tennis racket": 38,
  "bottle": 39,
  "wine glass": 40,
  "cup": 41,
  "fork": 42,
  "knife": 43,
  "spoon": 44,
  "bowl": 45,
  "banana": 46,
  "apple": 47,
  "sandwich": 48,
  "orange": 49,
  "broccoli": 50,
  "carrot": 51,
  "hot dog": 52,
  "pizza": 53,
  "donut": 54,
  "cake": 55,
  "chair": 56,
  "couch": 57,
  "potted plant": 58,
  "bed": 59,
  "dining table": 60,
  "toilet": 61,
  "tv": 62,
  "laptop": 63,
  "mouse": 64,
  "remote": 65,
  "keyboard": 66,
  "cell phone": 67,
  "microwave": 68,
  "oven": 69,
  "toaster": 70,
  "sink": 71,
  "refrigerator": 72,
  "book": 73,
  "clock": 74,
  "vase": 75,
  "scissors": 76,
  "teddy bear": 77,
  "hair drier": 78,
  "toothbrush": 79
}


# class for applying segmenting images
class Segmenter:
    def __init__(self, 
                 yolo_model_location,
                 labels=['bottle','cup'],
                 conf=0.25,
                 max_det=1):
        self.conf = conf
        self.model = YOLO(yolo_model_location)
        try:
            self.model.to('cuda')
        except:
            pass
        self.labels_text = labels
        self.labels_ints = [label_lookup_dict[v] for v in labels]
        self.max_det = max_det

        self.img = None
        self.img_height = None
        self.img_width = None
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None
        self.seg = None
        self.segmentation_result = None
        self.detection_window_path = parent_dir + '/temp/temp.png'

    def reset(self):
        self.img = None
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None
        self.seg = None
        self.width = None
        self.height = None
        self.segmentation_result = None

    def read_img_path(self, img_path):
        self.reset()
        self.img = clipper(resizer(read_image(img_path))).unsqueeze(0)
        shapes = self.img.shape
        h = shapes[3]
        w = shapes[2]
        self.height = h
        self.width = w

    def read_img(self, img):
        self.reset()
        self.img = img
        h, w, _ = self.img.shape
        self.height = h
        self.width = w

    def segment(self):
        self.segmentation_result = self.model.predict(source=self.img,
                                                      classes=self.labels_ints,
                                                      conf=self.conf,
                                                      show_labels=False,
                                                      boxes=False,
                                                      verbose=False,
                                                      half=True,
                                                      max_det=self.max_det)

        # class names
        self.class_names = self.model.names

        # random colors for plotting
        # self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.class_names]
        self.colors = [[100,0,100] for _ in self.class_names]

        # extract segmentation result
        batch, channels, h, w = self.img.shape
        boxes = self.segmentation_result[0].boxes
        masks = self.segmentation_result[0].masks

        # extract segmentation result
        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.data.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (w, h)).astype(np.uint8)

                self.xmin = int(box.data[0][0])
                self.ymin = int(box.data[0][1])
                self.xmax = int(box.data[0][2])
                self.ymax = int(box.data[0][3])
                self.seg = seg

                self.detection_window = self.img[self.ymin:self.ymax,
                                                 self.xmin:self.xmax]

                break

    def save_segment(self):
        cv2.imwrite(self.detection_window_path, self.detection_window)

    @staticmethod
    def overlay(image, mask, color, alpha, resize=None):
        colored_mask = np.stack((mask,)*3, axis=-1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()

        if resize is not None:
            image = cv2.resize(image.transpose(1, 2, 0), resize)
            image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

        image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

        return image_combined

    @staticmethod
    def plot_one_box(x, img, color=None, label=None, line_thickness=3):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def project_segmentations(self, show_overlay=True, show_boxes=False, show_result=False):
        # bring img back to cpu 
        self.img = self.img.cpu().numpy()
        self.img = np.squeeze(self.img).transpose((1, 2, 0))
        self.orig_img = self.img.copy()

        # unpack segmentation results
        boxes = self.segmentation_result[0].boxes
        masks = self.segmentation_result[0].masks

        # loop over masks and plot
        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.data.cpu().numpy(), boxes):   
                if show_overlay:
                    self.img = self.overlay(self.img, seg, self.colors[int(box.cls)], 0.4)

                xmin = int(box.data[0][0])
                ymin = int(box.data[0][1])
                xmax = int(box.data[0][2])
                ymax = int(box.data[0][3])

                if show_boxes:
                    self.plot_one_box([xmin, ymin, xmax, ymax],
                                      self.img,
                                      self.colors[int(box.cls)],
                                      f'{self.class_names[int(box.cls)]} {float(box.conf):.3}')

    def show_result(self):
        plt.imshow(self.img)
        plt.axis('off')  # optional: disable the axis
        plt.show()
        

# employ segmenter
def segment_image(img_path,
                  labels=['bottle','cup'],
                  conf=0.05,
                  max_det=1,
                  yolo_model_location=yolo_model_location):
  
    # create instance of Segmenter
    seg = Segmenter(yolo_model_location,
                    labels=labels,
                    conf=conf,
                    max_det=max_det)
    
    # run segment
    seg.read_img_path(img_path)
    seg.segment()
    seg.project_segmentations()

    # unpack image and mask from segmenter
    img = seg.img

    # extract mask - bring back to cpu
    mask = seg.segmentation_result[0].masks.data[0].cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask = np.stack((mask,) * 3, axis=2)

    return img, mask, seg
