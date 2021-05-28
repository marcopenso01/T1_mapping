"""
Created on Mon Feb 22 18:23:43 2021

@author: Marco Penso
"""

import os
import numpy as np
import logging
import cv2
import shutil
import pydicom 
import matplotlib.pyplot as plt
import math
import skimage.morphology, skimage.data
from PIL import Image

drawing=False # true if mouse is pressed
mode=True

def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False

def deletefolder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
        return True
    return False

# mouse callback function
def paint_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),3)
                cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),3)
                current_former_x = former_x
                current_former_y = former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),3)
            cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),3)
            current_former_x = former_x
            current_former_y = former_y
    return former_x,former_y

def imfill(img, size):
    img = img[:,:,0]
    img = cv2.resize(img, (size, size))
    img[img>0]=255
    im_floodfill = img.copy()
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    return img | cv2.bitwise_not(im_floodfill)

# click event function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x,",",y)
        X.append(y)
        Y.append(x)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    
    # leggo immagini
    path = r'F:\t1'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info('................................................')
    for paz in sorted(os.listdir(path)):
        logging.info('Processing: %s' % paz)
        paz_fold = os.path.join(path, paz)
        f=[os.path.join(paz_fold, 't1_img'), os.path.join(paz_fold, 'ct_img')]
        for i in range(len(f)):
            if not os.path.exists(f[i]):
                makefolder(f[i])
            if os.path.exists(f[i]):
                deletefolder(f[i])
            makefolder(f[i])    
        # t1
        for fold in os.listdir(os.path.join(paz_fold, 't1_dcm')):
            for file in sorted(os.listdir(os.path.join(paz_fold, 't1_dcm', fold))):
                fn = file.split('.dcm')
                dcmPath = os.path.join(paz_fold, 't1_dcm', fold, file)
                data_row_img = pydicom.dcmread(dcmPath)
                image = np.uint16(data_row_img.pixel_array)
                maxx = image.max()
                image = cv2.convertScaleAbs(image, alpha=(255.0/maxx))
                cv2.imwrite(os.path.join(f[0], fn[0] + '.png'), image)
        
        # ct
        for fold in os.listdir(os.path.join(paz_fold, 'tac_dcm')):
            for file in sorted(os.listdir(os.path.join(paz_fold, 'tac_dcm', fold))):
                fn = file.split('.dcm')
                dcmPath = os.path.join(paz_fold, 'tac_dcm', fold, file)
                data_row_img = pydicom.dcmread(dcmPath)
                image = data_row_img.pixel_array
                #array_buffer = image.tobytes()
                #img = Image.new("I", image.shape)
                #img.frombytes(array_buffer, 'raw', "I;16")
                #img.save(os.path.join(f[1], fn[0] + '.png'))
                image = image - image.min()
                maximum = image.max()
                grayscale = (image / maximum) * 255
                cv2.imwrite(os.path.join(f[1], fn[0] + '.png'), grayscale)
                


image_stacks = np.zeros(shape=(472, 512, 512), dtype=np.int16)
for file, i in zip(sorted(os.listdir(r'F:\t1\paz3\ScalarVolume_85')), range(len(image_stacks[0]))):
    path = os.path.join(r'F:\t1\paz3\ScalarVolume_85',file)
    fn = file.split('.dcm')
    data_row_img = pydicom.dcmread(path)
    image = data_row_img.pixel_array
    image_stacks[i] = np.array(image)

#save img
for i in range(image_stacks.shape[2]):
    img = image_stacks[:,:,i]
    img_new = (img - img.min()) * ((255 - 0) / (img.max() - img.min()))
    img_new = np.uint8(img_new)
    cv2.imwrite(os.path.join(r'F:\t1\paz3\prova' , 'IMG'+ str(i) + '.png'), cv2.flip(img_new, 0))

        

    '''
    # genero segmenti
    N = 7
    path = r'G:'
    force_overwrite = True
    t1_fold = os.path.join(path, 't1-map')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info('................................................')
    cv2.destroyAllWindows()
    mask_path = os.path.join(path, 'maschere_t1')            
    if not os.path.exists(mask_path):
        makefolder(mask_path)
    crop_path = os.path.join(path, 'crop_t1')            
    if not os.path.exists(crop_path):
        makefolder(crop_path)    
        
    for paz in sorted(os.listdir(t1_fold)):
        logging.info('Processing: %s' % paz)
        path_paz = os.path.join(t1_fold, paz)
        for file in sorted(os.listdir(path_paz)):
            addr_t1 = os.path.join(path_paz,file)
            img_t1 = np.uint8(pydicom.dcmread(addr_t1).pixel_array)
            #plt.imshow(img_t1)
            if force_overwrite:
                if os.path.exists(os.path.join(mask_path, paz)):
                    deletefolder(os.path.join(mask_path, paz))
                makefolder(os.path.join(mask_path, paz))
                if os.path.exists(os.path.join(crop_path, paz)):
                    deletefolder(os.path.join(crop_path, paz))
                makefolder(os.path.join(crop_path, paz))
            logging.info('---Cropping image---:')
            crop = cv2.selectROI(img_t1, showCrosshair = False)
            cv2.destroyAllWindows()
            if crop[2] < crop[3]:
                slice_cropped = img_t1[int(crop[1]):int(crop[1]+crop[3]), int(crop[0]):int(crop[0]+crop[3])]
            elif crop[2] > crop[3]:
                slice_cropped = img_t1[int(crop[1]):int(crop[1]+crop[2]), int(crop[0]):int(crop[0]+crop[2])]
            else:
                slice_cropped = img_t1[int(crop[1]):int(crop[1]+crop[3]), int(crop[0]):int(crop[0]+crop[2])]
            logging.info('crop: %s %s %s %s' % (crop[0],crop[1],crop[2],crop[3]))
            #plt.imshow(slice_cropped)
            cv2.imwrite(os.path.join(crop_path, paz, file.split('.dcm')[0] + '.png'), slice_cropped[:,:,0])
            logging.info('---select the point where the RV wall joins the LV')
            X = []
            Y = []
            cv2.imshow("image", slice_cropped)
            cv2.namedWindow('image')
            cv2.setMouseCallback("image", click_event)
            cv2.waitKey(0)
            logging.info('---Segmenting myocardium---:')
            tit=['endocardium contour', 'epicardium contour']
            for ii in range(2):
                img = np.array(slice_cropped)
                img = cv2.resize(img, (400, 400))
                image_binary = np.zeros((img.shape[1], img.shape[0], 1), np.uint8)
                cv2.namedWindow(tit[ii])
                cv2.setMouseCallback(tit[ii],paint_draw)
                while(1):
                    cv2.imshow(tit[ii],img)
                    k=cv2.waitKey(1)& 0xFF
                    if k==27: #Escape K
                        if ii==0:
                            im_out1 = imfill(image_binary, slice_cropped.shape[0])
                            im_out1[im_out1>0]=255
                            contours, _ = cv2.findContours(im_out1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                            ref = np.zeros_like(im_out1)
                            cv2.drawContours(ref, contours, 0, 255, 1);
                            M = cv2.moments(ref)
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                        elif ii==1:                 
                            im_out2 = imfill(image_binary, slice_cropped.shape[0])
                            im_out2[im_out2>0]=255
                        break              
                cv2.destroyAllWindows()
            #myo binary mask
            im_out = im_out2-im_out1
            plt.imshow(im_out)
            cv2.imwrite(os.path.join(mask_path, paz, file.split('.dcm')[0] + '.png'), im_out)
            #myo mask
            myo_mask = slice_cropped.copy()
            labels = skimage.morphology.label(im_out)
            labelCount = np.bincount(labels.ravel())
            background = np.argmax(labelCount)
            myo_mask[labels == background] = 0
            cv2.imwrite(os.path.join(mask_path, paz, 'mask_myo'+ '.png'), myo_mask)
            #histogram
            histogram, bin_edges = np.histogram(myo_mask, bins=256, range=(0, 255))
            plt.figure()
            x = 1  #min value to plot
            plt.xlim([x, 255]) 
            plt.plot(bin_edges[x:-1], histogram[x::])
            #a[(a >> 0) & (a <= 145)] = 30
            
            phi = round(np.rad2deg(math.atan2(cY - cY, X[0] - cX) - math.atan2(Y[0] - cY, X[0] - cX)))
            contours, _ = cv2.findContours(im_out2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            ref = np.zeros_like(im_out2)
            cv2.drawContours(ref, contours, 0, 255, 1);
            for i in range(N):
                tmp = np.zeros_like(im_out)
                mask_out = slice_cropped.copy()
                for ii in range(2):
                    theta = (i+ii)*(360/N)-90-phi
                    theta *= np.pi/180.0
                    cv2.line(tmp, (cX, cY),
                             (int(cX+np.cos(theta)*ref.shape[0]),
                              int(cY-np.sin(theta)*ref.shape[0])), 255, 1);
                tmp = tmp[..., np.newaxis]
                tmp = imfill(tmp, tmp.shape[0])
                labels = skimage.morphology.label(tmp)
                labelCount = np.bincount(labels.ravel())
                background = np.argmax(labelCount)
                tmp[labels != background] = 255
                tmp[labels == background] = 0
                #binary mask of the segment
                out = tmp & im_out
                #plt.imshow(out)
                cv2.imwrite(os.path.join(mask_path, paz, 'binary_mask' + str(i) + '.png'), out)
                #mask of the segment
                labels = skimage.morphology.label(out)
                labelCount = np.bincount(labels.ravel())
                background = np.argmax(labelCount)
                mask_out[labels == background] = 0
                cv2.imwrite(os.path.join(mask_path, paz, 'mask' + str(i) + '.png'), mask_out)
    '''       
