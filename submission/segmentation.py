import cv2
import numpy as np

TEST_RATIO_MODE = False
img_area = 0

# Calculate skew angle of an image

def awesomize(image):
    """Normalizes the image

    Args:
        image (np.array): image

    Returns:
        result_norm (np.array): normalized array
    """
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

def contrast(img):
    """Improves contrast to deal with blurry images

    Args:
        img (np.array): Image

    Returns:
        img2 (np.array): high-contrast image
    """
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe=cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

    lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l,a,b=cv2.split(lab)  # split on 3 different channels

    l2=clahe.apply(l)  # apply CLAHE to the L-channel

    lab=cv2.merge((l2,a,b))  # merge channels
    img2=cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img2

def getminAreaRect(cvImage):
    """Gets the Rectangle of Minimum Area

    Args:
        cvImage (np.array): input image

    Returns:
        minAreaRect (tuple): Information about the Rectangle of Minimum Area
    """
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()

    cvImage = awesomize(cvImage)

    gray = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)
    return minAreaRect

def getCroppedCoordinates(cvImage):
    """Gets Coordinates of the bounding rectangle

    Args:
        cvImage (np.array): image

    Returns:
        tuple: Information about the Bounding Rectangle
    """
    newImage = cvImage.copy()

    cvImage = awesomize(newImage)
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # cv2.imshow("blur", blur)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    x, y, w, h = cv2.boundingRect(largestContour)
    return (x, y, w, h)

def rotateImage(cvImage, angle: float):
    """Rotates the image according to angle provided

    Args:
        cvImage (np.array): Image to be rotated
        angle (float): Angle to be rotated

    Returns:
        newImage (np.array): Rotated Image
    """
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

def getSkewAngle(cvImage) -> float:
    """Gets angle to be rotated by

    Args:
        cvImage (np.array): Image to be rotated

    Returns:
        float: Angle to be rotated by
    """
    newImage = cvImage.copy()
    minAreaRect = getminAreaRect(newImage)
    box = np.int0(cv2.boxPoints(minAreaRect))
    cv2.drawContours(newImage, [box], -1, (0, 255, 0), 2)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    # print(angle)
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

def resize_with_aspect(image, width: int = 500):
    """Resize the image while preserving the aspect ratio

    Args:
        image (np.array): Image to be resized
        width (int, optional): Width to be resized to. Defaults to 500.

    Returns:
        np.array: Resized Image
    """
    height = int(width/image.shape[1] * image.shape[0])
    return cv2.resize(image, (width, height))

def remove_horiz_line(new_img):
    global TEST_RATIO_MODE, img_area

    new_img = awesomize(new_img)

    x_crop, y_crop, w_crop, h_crop = getCroppedCoordinates(new_img)
    new_img = new_img[y_crop: y_crop + h_crop, x_crop: x_crop + w_crop]
    gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
    img_area = w_crop * h_crop
    # minAreaRect = getminAreaRect(new_img)
    # box = np.int0(cv2.boxPoints(minAreaRect))
    # img_area = cv2.contourArea(box)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # cv2.imshow("thresh", thresh)
    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # print("Number of horizontal lines:", len(cnts))

    # Setting Ratio of the figure above which horizontal lines should be removed.
    RATIO_TO_BE_KEPT = 0.2
    y_lim = h_crop * RATIO_TO_BE_KEPT

    # Displaying minAreaRect
    thresh_copy = cv2.bitwise_not(thresh)
    thresh_copy = cv2.cvtColor(thresh_copy,cv2.COLOR_GRAY2BGR)
    # box = np.int0(cv2.boxPoints(minAreaRect))
    # cv2.drawContours(thresh_copy, [box], -1, (255, 255, 0), 2)
    # cv2.imshow("minRect", thresh_copy)

    if TEST_RATIO_MODE:
        cv2.drawContours(thresh_copy, [cnts[-1]], -1, (0, 255, 0), -1)
        # cv2.drawContours(thresh_copy, [cnts[-2]], -1, (255, 0, 0), -1)
        # cv2.drawContours(thresh_copy, [cnts[-3]], -1, (0, 0, 255), -1)
        # cv2.drawContours(thresh_copy, [cnts[-4]], -1, (0, 255, 0), -1)
        # cv2.drawContours(thresh_copy, [cnts[-5]], -1, (255, 0, 0), -1)
        # cv2.drawContours(thresh_copy, [cnts[-6]], -1, (0, 0, 255), -1)
        # cv2.drawContours(thresh_copy, [cnts[-7]], -1, (0, 255, 0), -1)
        # cv2.drawContours(thresh_copy, [cnts[-8]], -1, (255, 0, 0), -1)

    # print("minAreaRect info:", minAreaRect)
    # print("y\ty_lim")
    if len(cnts) != 0:
        cv2.drawContours(thresh, [cnts[-1]], -1, (0, 0, 0), -1)
    for c in cnts:
        y = cv2.boundingRect(c)[1]
        # print(y, "   {:.2f}".format(y_lim))
        if not TEST_RATIO_MODE:
            if y < y_lim:
                cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
    return thresh


def perform_segmentation(image):
    """Performs Segmentation

    Args:
        image (np.array) : cv2 image array

    Returns:
        list(np.array): List of segmented images
    """
    global TEST_RATIO_MODE, img_area

    # image = cv2.imread(path)

    # print(image.shape)

    image = cv2.copyMakeBorder(image, image.shape[0]//20, image.shape[0]//20, image.shape[1]//20, image.shape[1]//20, cv2.BORDER_REPLICATE)

    image = resize_with_aspect(image, 500)
    image = contrast(image)
    image = cv2.fastNlMeansDenoisingColored(image ,None,10,10,7,21)

    angle = getSkewAngle(image)
    new_img = rotateImage(image, -1.0 * angle)
    new_img = resize_with_aspect(new_img, 500)
    # cv2.imshow("new_img", new_img)
    # angle = getSkewAngle(image)
    # new_img = rotateImage(image, -1.0 * angle)


    iwl_bb = remove_horiz_line(new_img) # Image Without Line_Black Background
    iwl_wb = cv2.bitwise_not(iwl_bb) # Image Without Line_White Background

    if TEST_RATIO_MODE:
        exit()

    # print()
    # print("MinAreaRect Area:", img_area)

    contours,_=cv2.findContours(iwl_bb,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rectangles=[]
    # print("Number of contours:", len(contours))
    # print("Contour Area\tImage Area Limit")
    for cnt in contours:
        if cv2.contourArea(cnt) > img_area / 2:
            # print(cv2.contourArea(cnt), "\t\t  ", img_area/2)
            continue
        if cv2.contourArea(cnt) > img_area / 1000:
            # print(cv2.contourArea(cnt), "\t\t  ", img_area/1000)
            x,y,w,h=cv2.boundingRect(cnt)
            # print(h, iwl_bb.shape[0] / 4)
            if(h < (iwl_bb.shape[0] / 4)):
                continue
            # print(x, y, w, h)
            rectangles.append([x,y,w,h])

    rectangles.sort()
    print("Number of characters recognized:", len(rectangles))

    images = []
    for i in range(len(rectangles)):
        cv2.rectangle(iwl_wb,
            (rectangles[i][0],rectangles[i][1]),
            (rectangles[i][0]+rectangles[i][2],
            rectangles[i][1]+rectangles[i][3]),
            (0,255,0),
            3
        )
        image=iwl_bb[
            rectangles[i][1] : rectangles[i][1] + rectangles[i][3],
            rectangles[i][0] : rectangles[i][0] + rectangles[i][2]
        ]
        larger = max(image.shape[0], image.shape[1])
        smaller = min(image.shape[0], image.shape[1])
        border=int(0.2 * larger)
        image = cv2.copyMakeBorder(
            image,
            top=(larger - smaller) // 2 + border,
            bottom=(larger - smaller) // 2 + int( 1 * border),
            left=border,
            right=int(1*border),
            borderType=cv2.BORDER_CONSTANT,
            value=[0,0,0])
        images.append(image)
    return images


if __name__ == '__main__':
    path = "/content/drive/MyDrive/Mosaic1 sample/samay2.jpg"
    image = cv2.imread(path)
    images = perform_segmentation(image)
    print("shape ",len(images))