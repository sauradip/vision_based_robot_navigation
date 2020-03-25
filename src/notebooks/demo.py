

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import sys
sys.path.append('../')
from models import net

import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable

# Pre-processing and post-processing constants #
CMAP = np.load('../cmap_nyud.npy')
print(CMAP)
DEPTH_COEFF = 5000. # to convert into metres
HAS_CUDA = torch.cuda.is_available()
# HAS_CUDA = False
IMG_SCALE  = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
MAX_DEPTH = 8.
MIN_DEPTH = 0.
NUM_CLASSES = 40
NUM_TASKS = 2 # segm + depth

def prepare_img(img):
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD

model = net(num_classes=NUM_CLASSES, num_tasks=NUM_TASKS)
if HAS_CUDA:
    _ = model.cuda()
_ = model.eval()

ckpt = torch.load('../../weights/ExpNYUD_joint.ckpt')
model.load_state_dict(ckpt['state_dict'])

mask_img = np.zeros((480,640))
cap = cv2.VideoCapture(-1)
# print(torch.cuda.available())
xx = 0 
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    xx+=1
    # Figure 2-top row
    # img_path = '../../examples/ExpNYUD_joint/000464.png'
    # img_path = '../../img/000281.jpg'
    img_path = frame
    straight = 0
    right_turn=0
    left_turn = 0
    no_move = 1
    # img = np.array(Image.open(img_path))
    img = np.array(img_path)
    gt_segm = np.array(Image.open('../../examples/ExpNYUD_joint/segm_gt_000464.png'))
    gt_depth = np.array(Image.open('../../examples/ExpNYUD_joint/depth_gt_000464.png'))
    mask_img = np.zeros((480,640))
    with torch.no_grad():
        img_var = Variable(torch.from_numpy(prepare_img(img).transpose(2, 0, 1)[None]), requires_grad=False).float()
        if HAS_CUDA:
            img_var = img_var.cuda()
        segm, depth = model(img_var)
        kernel = np.ones((5,5), np.uint8)
        # img_erosion = cv2.erode(segm, kernel, iterations=1)
        segm = cv2.resize(segm[0, :NUM_CLASSES].cpu().data.numpy().transpose(1, 2, 0),
                        img.shape[:2][::-1],
                        interpolation=cv2.INTER_CUBIC)
        # print(np.shape(segm))
        depth = cv2.resize(depth[0, 0].cpu().data.numpy(),
                           img.shape[:2][::-1],
                           interpolation=cv2.INTER_CUBIC)
        # if segm.argmax(axis=2) == 1
        segm = CMAP[segm.argmax(axis=2) + 1].astype(np.uint8)
        # cv2.imwrite("img/"+str(xx)+".png",img)
        print(np.shape(segm)[0])
        depth = np.abs(depth)
        for i in range(0,img.shape[0]):
            for j in range(0,img.shape[1]):
                if round(depth[i][j]) <= 2 and segm[i,j,0] == 0 and segm[i,j,1]==128 and segm[i,j,2]==0 :
                    mask_img[i][j]=1
        # cv2.imwrite("depth/"+str(xx)+".png",depth)
        # cv2.imwrite("seg/"+str(xx)+".png",segm)
        h = img.shape[0] ### 480
        w = img.shape[1] ### 640
	## left mat will be 480 x 210 , right mat will be 480 x 210 and middle will be 480 x 220
        left_mat = mask_img[0:h,0:209]
        middle_mat = mask_img[0:h,210:429]
        right_mat = mask_img[0:h,430:639]

        left_nzero = np.count_nonzero(left_mat)
        right_nzero = np.count_nonzero(right_mat)
        mid_nzero = np.count_nonzero(middle_mat)

        if mid_nzero > right_nzero and mid_nzero > left_nzero :
            straight = 1
        elif right_nzero > mid_nzero and right_nzero > left_nzero :
            right_turn = 1
        elif left_nzero > mid_nzero and left_nzero > right_nzero :
            left_turn = 1
        else:
            no_move = 1
        
        

        
        cam_cen_x=round(h)
        cam_cen_y=round(w/2)
        img2 = np.zeros_like(img)
        img3 = np.array(cv2.merge((mask_img,mask_img,mask_img)))
        img3[img3 == 1 ] = 255

        print(img.dtype)
        print(img3.dtype)
        # img2[:,:,0] = mask_img
        # img2[:,:,1] = mask_img
        # img2[:,:,2] = mask_img
        # print(np.shape(img2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        img3[np.where((img3==[255,255,255]).all(axis=2))] = [0,128,0]
        img_mask_color = cv2.addWeighted(img, 0.5, img3.astype(np.uint8), 0.5, 1)

        if straight == 1:
            cv2.putText(img_mask_color, 'Move Straight', (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        elif right_turn == 1:
            cv2.putText(img_mask_color, 'Move Right', (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        elif left_turn == 1:
            cv2.putText(img_mask_color, 'Move Left', (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        else :
            cv2.putText(img_mask_color, 'Do Not Move', (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # cv2.line(mask_img, (cam_cen_y, cam_cen_x) , (round(w/2),round(h/2)), (0, 255, 0) , 8 ) ####### straight line ###
        # cv2.line(mask_img, (cam_cen_y-50, cam_cen_x) , (0,round(h/2)), (0, 255, 0) , 8 ) ####### diagonal line ###
        # cv2.line(mask_img, (cam_cen_y+50, cam_cen_x) , (w,round(h/2)), (0, 255, 0) , 8 ) ####### diagonal line ###

        # cv2.imwrite("fuse/"+str(xx)+".png",mask_img)
        # print(mask_img)
        # print(np.shape(depth))
        cv2.imshow('segmentation',img_mask_color)
    # cv2.imshow('input',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()  
# plt.figure(figsize=(18, 12))
# plt.subplot(151)
# plt.imshow(img)
# plt.title('orig img')
# plt.axis('off')
# plt.subplot(152)
# plt.imshow(CMAP[gt_segm + 1])
# plt.title('gt segm')
# plt.axis('off')
# plt.subplot(153)
# plt.imshow(segm)
# plt.title('pred segm')
# plt.axis('off')
# plt.subplot(154)
# plt.imshow(gt_depth / DEPTH_COEFF, cmap='plasma', vmin=MIN_DEPTH, vmax=MAX_DEPTH)
# plt.title('gt depth')
# plt.axis('off')
# plt.subplot(155)
# plt.imshow(depth, cmap='plasma', vmin=MIN_DEPTH, vmax=MAX_DEPTH)
# plt.title('pred depth')
# plt.axis('off');
