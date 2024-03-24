import argparse
import torch
from pre_data import *
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import time

parser = argparse.ArgumentParser(description='YOLO Detection')
parser.add_argument('-v', '--version', default='yolo',
                    help='yolo')
parser.add_argument('--video',default=0,type=str,help='the video path')
parser.add_argument('-size', '--input_size', default=416, type=int,
                    help='input_size')
parser.add_argument('--trained_model', default=r'.\weights\voc\yolo_64.4_68.5_71.5.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--conf_thresh', default=0.1, type=float,
                    help='Confidence threshold')
parser.add_argument('--nms_thresh', default=0.50, type=float,
                    help='NMS threshold')
parser.add_argument('--visual_threshold', default=0.3, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')

args = parser.parse_args()

def vis(img, bboxes, scores, cls_inds, thresh, class_colors, class_names):
    for i, box in enumerate(bboxes):
        cls_indx = cls_inds[i]
        xmin, ymin, xmax, ymax = box
        if scores[i] > thresh:
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 1)
            cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_colors[int(cls_indx)], -1)
            mess = '%s' % (class_names[int(cls_indx)])
            cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return img
        
def test_videos(net,device,video_path,transform,thresh,class_colors=None,class_names=None,alpha=False):
    video = cv2.VideoCapture(video_path)
    while True:
        begin = time.time()
        ret,frame = video.read()
        h,w,_ = frame.shape
        if alpha:
            x =torch.from_numpy(np.asarray(transform(frame)).astype(np.float32)[:, :, (2, 1, 0)]).permute(2, 0, 1)
        else:
            transform = BaseTransform(input_size)
            x =torch.from_numpy(transform(frame)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
        x = x.unsqueeze(0).to(device)
        
        bboxes, scores, cls_inds = net(x)

        scale = np.array([[w, h, w, h]])
        # map the boxes to origin image scale
        bboxes *= scale

        img_processed = vis(frame, bboxes, scores, cls_inds, thresh, class_colors, class_names)

        exec_time = time.time()-begin
        curr_fps = 10/exec_time
        fps = "FPS: " + str(curr_fps)

        cv2.putText(img_processed, fps, (100,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.imshow('detection', img_processed)
        if cv2.waitKey(1) & 0xFF == 27 :
            break 

if __name__ == '__main__':
    # get device
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    input_size = [args.input_size, args.input_size]

    class_colors = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(VOC_CLASSES_NUM)]

    # build model
    if args.version == 'yolo':
        from models.yolo import myYOLO
        net = myYOLO(device, input_size=input_size, num_classes=VOC_CLASSES_NUM, trainable=False)

    else:
        exit()

    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.to(device).eval()
    print('Finished loading model!')

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size)  
    ])
    test_videos(
        net=net,
        device=device,
        video_path=0,
        transform=transform,
        thresh=args.visual_threshold,
        class_colors=class_colors,
        class_names=VOC_CLASSES,
        alpha=False
    )
