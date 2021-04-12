'''
This file contains two functions:

1. predict: You will be given an rgb image which you will use to predict the output 
which will be a string. For the prediction you can use/import code,models from other files or
libraries. More detailes given above the function defination.

2. test: This will be used by the co-ordinators to test your code by giving sample 
inputs to the 'predict' function mentioned above. A sample test function is given for your
reference but it is subject to minor changes during the evaluation. However, note that
there won't be any changes in the input format given to the predict function.

Make sure all the necessary functions etc. you import are done from the same directory. And in 
the final submission make sure you provide them also along with this script.
'''

# Essential libraries and your model can be imported here
import os
import cv2
import numpy as np 

from predict import predict_charac
from segmentation import perform_segmentation

INV_MAP = {
    1 : 'क', 
    2 : 'घ ', 
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

'''
function: predict
input: image - A numpy array which is an rgb image
output: answer - A list which is the full word

Suggestion: Try to make your code understandable by including comments and organizing it. For 
this we encourgae you to write essential function in other files and import them here so that 
the final code is neat and not too big. Make sure you use the same input format and return 
same output format.
'''
def predict(image):
    '''
    Write your code for prediction here.
    '''
    imgs = perform_segmentation(image)#perfom segmenation and returns list of characters
    answer = []
    for img in imgs:#iterate over the character and make prediction
        preds = predict_charac(img)
        answer.append(INV_MAP[preds.item()])#append prediction
    return answer


'''
function: test
input: None
output: None

This is a sample test function which the co-ordinaors will use to test your code. This is
subject to change but the imput to predict function and the output expected from the predict
function will not change. 
You can use this to test your code before submission: Some details are given below:
image_paths : A list that will store the paths of all the images that will be tested.
correct_answers: A list that holds the correct answers
score : holds the total score. Keep in mind that scoring is subject to change during testing.

You can play with these variables and test before final submission.
'''
def test():
    '''
    We will be using a similar template to test your code
    '''
    image_paths = ['/content/drive/MyDrive/Mosaic1 sample/long word.jpeg']
    correct_answers = [["स" ,"म" ,"य"]]
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
                
        if(n==len(correct_answers[i])):
            score += len(correct_answers[i])*multiplication_factor

        else:
            score += n*2
        
    
    print('The final score of the participant is',score)


if __name__ == "__main__":
    test()