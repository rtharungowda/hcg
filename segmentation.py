import cv2
import numpy as np
import sys, os

TEST_RATIO_MODE = False
img_area = 0

# Calculate skew angle of an image

def awesomize(image):
    rgb_planes = cv2.split(image)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm

def getminAreaRect(cvImage):
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()

    cvImage = awesomize(cvImage)

    gray = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)
    return minAreaRect

def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

def getSkewAngle(cvImage) -> float:
    newImage = cvImage.copy()
    minAreaRect = getminAreaRect(newImage)
    box = np.int0(cv2.boxPoints(minAreaRect))
    cv2.drawContours(newImage, [box], -1, (0, 255, 0), 2)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    print(angle)
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

def resize_with_aspect(image, width: int = 500):
    height = int(width/image.shape[1] * image.shape[0])
    return cv2.resize(image, (width, height))

def remove_horiz_line(new_img):
    global TEST_RATIO_MODE, img_area

    new_img = awesomize(new_img)

    gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
    minAreaRect = getminAreaRect(new_img)
    box = np.int0(cv2.boxPoints(minAreaRect))
    img_area = cv2.contourArea(box)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    print("Number of horizontal lines:", len(cnts))

    # Setting Ratio of the figure above which horizontal lines should be removed.
    RATIO_TO_BE_KEPT = 0.18
    y_lim = minAreaRect[0][1] - minAreaRect[1][1] * (0.5 - RATIO_TO_BE_KEPT)

    # Displaying minAreaRect
    thresh_copy = cv2.bitwise_not(thresh)
    thresh_copy = cv2.cvtColor(thresh_copy,cv2.COLOR_GRAY2BGR)
    box = np.int0(cv2.boxPoints(minAreaRect))
    cv2.drawContours(thresh_copy, [box], -1, (255, 255, 0), 2)
    # cv2.imshow("minRect", thresh_copy)

    if TEST_RATIO_MODE:
        cv2.drawContours(thresh_copy, [cnts[-1]], -1, (0, 255, 0), 7)
        cv2.drawContours(thresh_copy, [cnts[-2]], -1, (255, 0, 0), 7)

    print("minAreaRect info:", minAreaRect)
    print("y\ty_lim")
    if len(cnts) != 0:
        cv2.drawContours(thresh, [cnts[-1]], -1, (0, 0, 0), 7)
    for c in cnts:
        y = cv2.boundingRect(c)[1]
        print(y, "   {:.2f}".format(y_lim))
        if not TEST_RATIO_MODE:
            if y < y_lim:
                cv2.drawContours(thresh, [c], -1, (0, 0, 0), 7)
    return thresh

def get_dominant_colour(image):
    data = np.reshape(image, (-1,3))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, _, centers = cv2.kmeans(data,1,None,criteria,10,flags)

    return centers[0].astype(int).tolist()

def perform_segmentation(image):
    global TEST_RATIO_MODE, img_area
    print(image.shape)
    image = cv2.copyMakeBorder(image, int(image.shape[0]/20), int(image.shape[0]/20), int(image.shape[1]/20), int(image.shape[1]/20), cv2.BORDER_REPLICATE)

    image = resize_with_aspect(image, 500)
    image = cv2.fastNlMeansDenoisingColored(image ,None,10,10,7,21)

    angle = getSkewAngle(image)
    new_img = rotateImage(image, -1.0 * angle)
    new_img = resize_with_aspect(new_img, 500)

    iwl_bb = remove_horiz_line(new_img) # Image Without Line_Black Background
    iwl_wb = cv2.bitwise_not(iwl_bb) # Image Without Line_White Background

    if TEST_RATIO_MODE:
        exit()

    print()

    # img_area = iwl_bb.shape[1] * iwl_bb.shape[0]
    # print(iwl_bb.shape[1], iwl_bb.shape[0])
    print("MinAreaRect Area:", img_area)

    contours,_=cv2.findContours(iwl_bb,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rect=[]
    mids=[]
    print("Number of contours:", len(contours))
    print("Contour Area\tImage Area Limit")
    for cnt in contours:
        if cv2.contourArea(cnt) > img_area / 2:
            # print(cv2.contourArea(cnt), "\t\t  ", img_area/2)
            continue
        if cv2.contourArea(cnt) > img_area / 210:
            print(cv2.contourArea(cnt), "\t\t  ", img_area/210)
            x,y,w,h=cv2.boundingRect(cnt)
            # print(x, y, w, h)
            rect.append([x,y,w,h])
            mids.append([x+w/2,y+h/2])

    rect.sort()
    print("Number of characters recognized:", len(rect))

    images = []
    for i in range(len(rect)):
        cv2.rectangle(iwl_wb,(rect[i][0],rect[i][1]),(rect[i][0]+rect[i][2],rect[i][1]+rect[i][3]),(0,255,0),3)
        image=iwl_bb[rect[i][1]:rect[i][1]+rect[i][3],rect[i][0]:rect[i][0]+rect[i][2]]
        if image.shape[1]>image.shape[0]:
            borderless=int(0.2*image.shape[1])
            image = cv2.copyMakeBorder(
                image,
                top=(image.shape[1]-image.shape[0])//2+borderless,
                bottom=(image.shape[1]-image.shape[0])//2+int(1*borderless),
                left=borderless,
                right=int(1*borderless),
                borderType=cv2.BORDER_CONSTANT,
                value=[0,0,0])
        else:
            borderless=int(0.2*image.shape[0])
            image = cv2.copyMakeBorder(
                image,
                left=(image.shape[0]-image.shape[1])//2+borderless,
                right=(image.shape[0]-image.shape[1])//2+int(1*borderless),
                top=borderless,
                bottom=int(1*borderless),
                borderType=cv2.BORDER_CONSTANT,
                value=[0,0,0])
        cv2.namedWindow("img_{}".format(i), cv2.WINDOW_GUI_EXPANDED)
        images.append(image)
    return images

    # TODO: Bounding box is not aligned properly after rotation? Maybe due to resize

if __name__ == '__main__':
    image = cv2.imread(os.path.join(os.getcwd(), filename))
    perform_segmentation(image)