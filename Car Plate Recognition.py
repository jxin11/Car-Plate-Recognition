import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
from numpy import diff
import functools
from PIL import Image, ImageOps
from PIL import ImageEnhance
import math
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import pytesseract
import time

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Define constants for Character Recognition
TARGET_WIDTH = 128    # 128
TARGET_HEIGHT = 128   # 128
chars = [
    '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G',
    'H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
]

# Load the pre-trained convolutional neural network
MODEL_PATH = './trained_model'
model = load_model(MODEL_PATH, compile=False)

start_time_LPLS = time.time()

# Import image
i = cv2.imread('TEST IMAGES\MCG7722.png')

# equalize the histogram of the Y channel
img_yuv = cv2.cvtColor(i, cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
i = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# Convert to HSV
hsv_img = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)

# region - Setting HSV Range for 5 Colours
sensitivity = 50
lower_white = np.array([0,0,255-sensitivity])
upper_white = np.array([179,sensitivity,255])

lower_black = np.array([0,0,50])
upper_black = np.array([179,50,255])  # 179 50 130

lower_blue = np.array([91,158,0])   # 91 158 0
upper_blue = np.array([138,255,255])  # 138 255 255

lower_yellow = np.array([20,100,0])
upper_yellow = np.array([35,255,255])

lower_green = np.array([40, 100, 0])
upper_green = np.array([90, 255, 255])
# endregion

# region - Colour Segmentiation

# Threshold the HSV image to get only X colors
mask_white = cv2.inRange(hsv_img, lower_white, upper_white)
mask_black = cv2.inRange(hsv_img, lower_black, upper_black)
mask_blue = cv2.inRange(hsv_img, lower_blue, upper_blue)
mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
mask_green = cv2.inRange(hsv_img, lower_green, upper_green)

# Bitwise-AND mask and original image
res_white = cv2.bitwise_and(i, i, mask=mask_white)
res_black = cv2.bitwise_and(i, i, mask=mask_black)
res_blue = cv2.bitwise_and(i, i, mask=mask_blue)
res_yellow = cv2.bitwise_and(i, i, mask=mask_yellow)
res_green = cv2.bitwise_and(i, i, mask=mask_green)

# For black
# ret, mask_black = cv2.threshold(cv2.cvtColor(i,cv2.COLOR_BGR2GRAY), 120, 255, cv2.THRESH_BINARY_INV)
# res_black = cv2.bitwise_and(cv2.bitwise_not(i), cv2.bitwise_not(i), mask=mask_black)

# endregion

# Morphological Opr on Each Color Extraction
res = [res_white, res_black, res_blue, res_yellow, res_green]
res_morph = []
for idx, r in enumerate(res):
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)

    ret, thresh = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    # cv2.imshow('thresh',thresh)
    
    # Noise removal using Morphological
    # closing operation
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                                kernel, iterations = 1)
    # cv2.imshow('closing',closing)
    
    # Background area using Dilation
    bg = cv2.dilate(closing, kernel, iterations = 2)
    # cv2.imshow('bg',bg)
    
    # Finding foreground area
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
    ret, fg = cv2.threshold(dist_transform, 0.01
                            * dist_transform.max(), 255, 0)
    # cv2.imshow('fg',fg)
    
    # Smoothing
    kernel = np.ones((3,3),np.float32)/5
    dst = cv2.filter2D(fg,-1,kernel)

    # Closing
    kernel = np.ones((3,3),np.float32)/5
    closing = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel, iterations=5)
    # cv2.imshow('closing2',closing)

    res_morph.append(dst)

# Draw Contour & Filter based on Size and Shape
newContours = []
colour = ['white', 'black', 'blue', 'yellow', 'green']
for idx, r in enumerate(res_morph):

    ## Remove Noise - bilateral filtering (blurring)
    # bf = cv2.bilateralFilter(r, 9, 20, 20)

    ## Edge Detection - Canny Edge Method
    # Min & Max Threshold Values
    # edge = cv2.Canny(r.astype(np.uint8), 280, 300)   # 30 200

    # Looking for contour
    contours = cv2.findContours(r.copy().astype(np.uint8),cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
    # cv2.imshow("contour", r)

    # Draw Contours
    draw_image = cv2.drawContours(i.copy(), contours, -1, (0,255,0), 2)
    # cv2.imshow("edge", draw_image)

    # Filter right contours
    minContourArea = i.shape[0]*i.shape[1]*0.005    # 0.005
    minCropArea = 800   #800
    maxWHratio = 5; minWHratio = 2.2     # 5 3
    
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if (w>h) and (cv2.contourArea(c)>=minContourArea) and (cv2.contourArea(c)>=minCropArea) and (x!=0 and y!=0 and x+w!=i.shape[0] and y+h!=i.shape[1]) and (w/h>=minWHratio) and (w/h<=maxWHratio):
    
            # temp = i.copy()
            # cv2.drawContours(temp, [c], -1, (0,255,0), 2)
            # cv2.imshow("contour", temp)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Masking the part other than the number plate
            mask = np.zeros(r.shape, np.uint8)
            new_image = cv2.drawContours(mask, [c], 0, (255), -2, )
            new_image = cv2.bitwise_and(i,i,mask=mask)

            # Crop Image
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            cropped = i[topx:bottomx+1, topy:bottomy+1]

            newContours.append(cropped)
            # cv2.imshow("cropped", cropped)

# 2nd Candidate Selection: Based on Change of Intensity
if len(newContours)>1:

    avgDiff = []
    for cropped in newContours:

        # Convert to gray -> plot intensity
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
        # bg=cv2.morphologyEx(cropped, cv2.MORPH_DILATE, se)
        # cropped_gray=cv2.divide(cropped, bg, scale=255)
        # cropped=cv2.threshold(cropped_gray, 0, 255, cv2.THRESH_OTSU )[1] 

        # kernel = np.ones((3,3),np.float32)/5
        # cropped = cv2.filter2D(cropped,-1,kernel)
        # cropped = cv2.equalizeHist(cropped)
        # ret, cropped = cv2.threshold(cropped, 0, 255,
        #                              cv2.THRESH_BINARY_INV +
        #                              cv2.THRESH_OTSU)
        
        w,h = cropped.shape
        buffer = int(w*0.1)
        c = cv2.line(cropped.copy(), (0,int(w/2)), (h,int(w/2)), (0, 255, 0),2) # centre
        c = cv2.line(c, (0,int(w/2)-buffer), (h,int(w/2)-buffer), (0, 255, 0),2) # upper
        c = cv2.line(c, (0,int(w/2)+buffer), (h,int(w/2)+buffer), (0, 255, 0),2) # lower

        upper = cropped[int(w/2)-buffer:int(w/2)+1-buffer, 0:h]
        centre = cropped[int(w/2):int(w/2)+1, 0:h]
        lower = cropped[int(w/2)+buffer:int(w/2)+1+buffer, 0:h]

        # Changes of intensity: Differentiation
        dydx_upper = diff(np.array(upper.transpose().flatten(),dtype=np.double))/diff(range(upper.shape[1]))
        dydx_upper_avg = sum(np.absolute(dydx_upper))/len(dydx_upper)
        # print ("dydx upper: ", dydx_upper_avg)

        dydx_centre = diff(np.array(centre.transpose().flatten(),dtype=np.double))/diff(range(centre.shape[1]))
        dydx_centre_avg = sum(np.absolute(dydx_centre))/len(dydx_centre)
        # print ("dydx centre: ", dydx_centre_avg)

        dydx_lower = diff(np.array(lower.transpose().flatten(),dtype=np.double))/diff(range(lower.shape[1]))
        dydx_lower_avg = sum(np.absolute(dydx_lower))/len(dydx_lower)
        # print ("dydx lower: ", dydx_lower_avg)
        
        # print ("avg dydx: ", (dydx_lower_avg+dydx_centre_avg+dydx_upper_avg)/3, "\n")
        avgDiff.append((dydx_lower_avg+dydx_centre_avg+dydx_upper_avg)/3)

        # Plotting
        # cv2.imshow("cropped", c)
        # # plt.subplot(411); plt.plot(range(centre.shape[1]),centre.transpose())
        # # ax1 = plt.subplot(411); plt.imshow(c, 'gray'); plt.setp(ax1.get_xticklabels(), visible=False)
        # plt.figure(figsize=(5, 3), dpi=80)
        # ax2 = plt.subplot(412); plt.plot(range(1,upper.shape[1]),dydx_upper); plt.title("Change of Intensity - Upper");plt.setp(ax2.get_xticklabels(), visible=False)  #dydx_upper
        # ax3 = plt.subplot(413); plt.plot(range(1,centre.shape[1]),dydx_centre); plt.title("Change of Intensity - Centre");plt.setp(ax3.get_xticklabels(), visible=False)
        # ax4 = plt.subplot(414); plt.plot(range(1,lower.shape[1]),dydx_lower); plt.title("Change of Intensity - Lower")
        # plt.show()      
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    targetLPidx = avgDiff.index(max(avgDiff))

else:
    targetLPidx = 0

try:
    targetLP = newContours[targetLPidx]
    finaltargetLP = targetLP.copy()
    
    print("---LPLS: %s seconds ---" % (time.time() - start_time_LPLS))

    # close on ESC key
    if cv2.waitKey(0) == 27: 
        cv2.destroyAllWindows()
        plt.close('all')
    
except IndexError:
    print("Fail to Detect Licence Plate.")
    quit()


# region - Plot: LP Localization & Segmentation

# cv2.imshow('frame',i)
# cv2.imshow('mask - white', mask_white)
# cv2.imshow('mask - black', mask_black)
# cv2.imshow('mask - blue', mask_blue)
# cv2.imshow('res-white', res_white)
# cv2.imshow('res-black', res_black)
# cv2.imshow('res-blue', res_blue)
# cv2.imshow('res-yellow', res_yellow)
# cv2.imshow('res-green', res_green)
# for i in range(len(res_morph)):
#     cv2.imshow(str(i), res_morph[i])
# cv2.imshow('Licence Plate Segmentation', newContours[targetLPidx])

#endregion


# Character Segmentation ####################

# Rotate // Deskew Image ##################
# ref: https://stackoverflow.com/questions/61791146/how-to-continue-processing-the-license-plate-crop?answertab=votes#tab-top

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def compute_skew(src_img):

    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print('upsupported image type')

    img = cv2.medianBlur(src_img, 3)

    edges = cv2.Canny(img,  threshold1 = 30,  threshold2 = 100, apertureSize = 3, L2gradient = True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w / 4.0, maxLineGap=h/4.0)
    angle = 0.0
    nlines = lines.size

    #print(nlines)
    cnt = 0
    for x1, y1, x2, y2 in lines[0]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        #print(ang)
        if math.fabs(ang) <= 30: # excluding extreme rotations
            angle += ang
            cnt += 1

    if cnt == 0:
        return 0.0
    return (angle / cnt)*180/math.pi

def deskew(src_img):
    return rotate_image(src_img, compute_skew(src_img))

# Image Enhancement

def colorReduction(img):
    m = np.argmax(img, axis=2)
    choices = [[255,255,255], [0,0,0], [0,255,0], [0,255,0], [0,0,255]]
    res = np.choose(m[...,np.newaxis],choices)
    return np.uint8(res)

# Ref: https://stackoverflow.com/questions/5906693/how-to-reduce-the-number-of-colors-in-an-image-with-opencv/59978096#59978096
def kmeans_color_quantization(image, clusters=2, rounds=5):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

start_time_CS = time.time()

# Deskew
targetLP = deskew(targetLP)
skewtargetLP = targetLP.copy()

# Fast mean denoising
# targetLP = cv2.fastNlMeansDenoisingColored(targetLP,None,15,15,1,1)

# Gamma Correction
gamma = 0.8    
invGamma = 1.0/gamma
table = np.array([((i / 255.0) ** invGamma) * 255
                for i in np.arange(0, 256)]).astype("uint8")
targetLP = cv2.LUT(targetLP, table)

# Increase Contrast
img3_PIL = Image.fromarray(targetLP) 
enh_con = ImageEnhance.Contrast(img3_PIL)
contrast = 1.2
img3_c = enh_con.enhance(contrast)
targetLP = np.asarray(img3_c)   # For reversing the operation

# targetLP = colorReduction(targetLP)
targetLP = kmeans_color_quantization(targetLP)

# Apply Gaussian blurring and thresholding 
# to reveal the characters on the license plate
gray = cv2.cvtColor(targetLP, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
thresh_ori = cv2.threshold(blurred, 0, 255,
	                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

## closing operation
# kernel = np.ones((3, 3), np.uint8)
# closing = cv2.morphologyEx(thresh_ori, cv2.MORPH_CLOSE,
#                             kernel, iterations = 1)

# Prepare for reverse, FG is black instead of white
thresh_inverse = cv2.bitwise_not(thresh_ori)

# for thresh in [thresh_inverse]:
for thresh in [thresh_ori, thresh_inverse]:
    # Perform connected components analysis on the thresholded image and
    # initialize the mask to hold only the components we are interested in
    _, labels = cv2.connectedComponents(thresh)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # Set lower bound and upper bound criteria for characters
    total_pixels = targetLP.shape[0] * targetLP.shape[1]
    lower = total_pixels // 110 # heuristic param, can be fine tuned if necessary
    upper = total_pixels // 10 # heuristic param, can be fine tuned if necessary

    # Loop over the unique components
    for (idx, label) in enumerate(np.unique(labels)):
        # If this is the background label, ignore it
        if label == 0:
            continue
    
        # Otherwise, construct the label mask to display only connected component
        # for the current label
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
    
        # If the number of pixels in the component is between lower bound and upper bound, 
        # add it to our mask
        if numPixels > lower and numPixels < upper:
            mask = cv2.add(mask, labelMask)

    # Find contours and get bounding box for each contour
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    # Sort the bounding boxes from left to right, top to bottom
    # sort by Y first, and then sort by X if Ys are similar
    def compare(rect1, rect2):
        if abs(rect1[1] - rect2[1]) > 10:
            return rect1[1] - rect2[1]
        else:
            return rect1[0] - rect2[0]

    boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare))
    a = targetLP.copy()

    # Character Height > Width
    boundingBoxes = [b for b in boundingBoxes if b[3] > b[2]]

    start_time_CR = time.time()
    pytesseract_vehicle_plate = []
    alexnet_vehicle_plate = []
    for box in boundingBoxes:
        x,y,w,h = box
        if h>w:
            
            # Crop the character from the mask
            # and apply bitwise_not because in our training data for pre-trained model
            # the characters are black on a white background
            croppedChar = mask[y:y+h, x:x+w]
            cv2.rectangle(a, (x,y), (x+w,y+h), (0, 255, 0), 2)
                        
            croppedChar = cv2.bitwise_not(croppedChar)

            # cv2.imshow('char_cropped_bitwise',croppedChar)

            # Remove noise
            # croppedChar = cv2.medianBlur(croppedChar,3)

            # Thresholding
            # croppedChar = cv2.threshold(croppedChar, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # Sharpening
            # kernel = np.array([[-1,-1,-1], [-1,11,-1], [-1,-1,-1]])
            # croppedChar = cv2.filter2D(croppedChar, -1, kernel)

            # Opening: Erosion followed by Dilation
            # kernel = np.ones((3,3),np.uint8)
            # croppedChar = cv2.morphologyEx(croppedChar, cv2.MORPH_OPEN, kernel)

            # cv2.imshow('char_blur_opening',croppedChar)

            # Get the number of rows and columns for each cropped image
            # and calculate the padding to match the image input of pre-trained model
            rows = croppedChar.shape[0]; columns = croppedChar.shape[1]
            paddingY = (TARGET_HEIGHT - rows) // 18 if rows < TARGET_HEIGHT else int(0.17 * rows)
            paddingX = (TARGET_WIDTH - columns) // 18 if columns < TARGET_WIDTH else int(0.45 * columns)

            # Apply padding to make the image fit for neural network model
            croppedChar = cv2.copyMakeBorder(croppedChar, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)

            #Character Recognition with pytesseract
            text = pytesseract.image_to_string(croppedChar, lang='eng',
                                               config="-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c page_separator='' --psm 10 --oem 1")
            if len(text.strip())==0:  pytesseract_vehicle_plate += ' '
            else: pytesseract_vehicle_plate += text.strip()
            
            # Convert and resize image
            croppedChar = cv2.cvtColor(croppedChar, cv2.COLOR_GRAY2RGB)     
            croppedChar = cv2.resize(croppedChar, (TARGET_WIDTH, TARGET_HEIGHT))

            # cv2.imshow('char_padding_tesseract',croppedChar)
            
            # Prepare data for prediction
            croppedChar = croppedChar.astype("float") / 255.0
            croppedChar = img_to_array(croppedChar)
            croppedChar = np.expand_dims(croppedChar, axis=0)

            # Make prediction
            prob = model.predict(croppedChar)[0]
            idx = np.argsort(prob)[-1]
            alexnet_vehicle_plate += chars[idx]

            
            # cv2.putText(a, chars[idx], (x,y+15), 0, 0.5, (0, 0, 255), 2)
            # if cv2.waitKey(0) == 27: 
            #     cv2.destroyAllWindows()

    
    # PLOT ######
    plt.subplot(2, 3, 1), plt.imshow(cv2.cvtColor(i,cv2.COLOR_BGR2RGB)); plt.title('Original'); plt.axis('off')
    plt.subplot(2, 3, 2), plt.imshow(cv2.cvtColor(finaltargetLP,cv2.COLOR_BGR2RGB)); plt.title('LP'); plt.axis('off')
    plt.subplot(2, 3, 3), plt.imshow(cv2.cvtColor(skewtargetLP,cv2.COLOR_BGR2RGB)); plt.title('Deskew LP'); plt.axis('off')
    plt.subplot(2, 3, 4), plt.imshow(cv2.cvtColor(a,cv2.COLOR_BGR2RGB)); plt.title('Char Segmentation'); plt.axis('off')
    plt.subplot(2, 3, 5), plt.text(0.5, 0.5,''.join(pytesseract_vehicle_plate), fontsize=15); plt.title('pytesseract');plt.axis('off')
    plt.subplot(2, 3, 6), plt.text(0.5, 0.5,''.join(alexnet_vehicle_plate), fontsize=15); plt.title('alexnet');plt.axis('off')
    plt.show()

    # print("[pyTesseract] Vehicle Car Plate Number: ", ''.join(pytesseract_vehicle_plate))
    # print("[AlexNet] Vehicle Car Plate Number: ", ''.join(alexnet_vehicle_plate))

    # close on ESC key
    if cv2.waitKey(0) == 27: 
        cv2.destroyAllWindows()
        plt.close('all')

    # print("---CR: %s seconds ---" % (time.time() - start_time_CR))