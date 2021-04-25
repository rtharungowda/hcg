import os
import cv2
import numpy as np 

from predict import predict_charac
from segmentation import perform_segmentation

INV_MAP = {
    1 : 'क', 
    2 : 'घ', 
    3 : 'च', 
    4 : 'छ', 
    5 : 'ज', 
    6 : 'झ',
    7 : 'ञ',
    8 : 'ट',
    9 : 'ठ', 
    10: 'ड', 
    11: 'त', 
    12: 'द', 
    13: 'न', 
    14: 'प', 
    15: 'फ', 
    16: 'ब', 
    17: 'म', 
    18: 'य', 
    19: 'र', 
    20: 'ल', 
    21: 'व', 
    22: 'ष', 
    23: 'स', 
    24: 'ह', 
    25: 'क्ष', 
    26: 'त्र', 
}


def predict(image):
    '''
    function: predict
    input: image - A numpy array which is an rgb image
    output: answer - A list which is the full word
    '''
    imgs = perform_segmentation(image)#perfom segmenation and returns list of characters
    answer = []
    for img in imgs:#iterate over the character and make prediction
        preds = predict_charac(img)
        answer.append(INV_MAP[preds.item()])#append prediction
    return answer

def test():
    '''
    function: test
    input: None
    output: None
    '''
    image_paths = [os.path.join(os.path.dirname(__file__), 'test_images', 'tagachanaba.jpg')]
    correct_answers = [["त" ,"घ" ,"छ", "न", "ब"]]
    score = 0
    multiplication_factor=2 #depends on character set size

    for i,image_path in enumerate(image_paths):
        image = cv2.imread(image_path) # This input format wont change
        answer = predict(image) # a list is expected
        print(''.join(answer))# will be the output string

        n=0
        for j in range(len(answer)):
            if correct_answers[i][j] == answer[j]:
                n+=1
    
    print('number of correctly predicted characters ',n)


if __name__ == "__main__":
    test()