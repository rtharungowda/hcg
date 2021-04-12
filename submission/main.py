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
    answer = ['क','ख','ग'] # sample needs to be modified
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
    image_paths = ['./image1','./image2',',./imagen']
    correct_answers = [list1,list2,listn]
    score = 0
    multiplication_factor=2 #depends on character set size

    for i,image_path in enumerate(image_paths):
        image = cv2.imread(image_path) # This input format wont change
        answer = predict(image) # a list is expected
        print(''.join(answer))# will be the output string

        if correct_answers[i] == answer:
            score += len(answer)*multiplication_factor
    
    print('The final score of the participant is',score)


if __name__ == "__main__":
    test()