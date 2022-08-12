# Real time gender and emotion recognition using CNNs.
The purpose of this project is to train Convolutional Neural Networks (CNNs) to 
recognize the gender and the emotion of a person from his face. The trained CNNs are used for real-time recognition using computer's webcam.

## A brief description of the project
Two CNNs are trained: one CNN for gender recognition ([dataset for gender recognition](https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset)) 
and the other one for emotion recognition ([dataset for emotion recognition](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)). 
The datasets are downloaded using the kaggle API.

Recognizable emotions are: angry,happy,neutral,sad and surprised. The emotion dataset also contain images for "disgusted" and "feaful", we have decided not to consider these two
emotions to make the task easier for the CNN (looking at some other notebooks it is clear that these two emotions are difficult to recognize).

Once the two CNNs are ready we want to use them for real time recognition using the computer's webcam. Before doing this we have to detect, frame by frame, if there are faces 
in our real time video. That's for two reasons:

1. if there are no faces, we will not pass any image to our CNNs and the video can continue with any classification.
2. if a face is detected, we don't pass the whole frame to the CNNS but only the portion of frame within which the algorithm has recognized to be a face.

In order to do this we have used the [haar-cascade-classifier](https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml) for frontal face
detection algorithm. This algorithm returns, frame by frame, the portion of frame within which there is a face: this is the image we will pass to the CNNs.

## Packages
* numpy
* tensorflow
* opencv

## Files description
* *gender_recognition* - a IPYNB file where the CNN for gender recognition is trained
* *gender_vgg16model.h5* - the resulting CNN from *gender_recognition* (accuracy: 93%)
* *emotion_recognition* - a IPYNB file where the CNN for emotion recognition is trained
* *my_emotion_model.h5* - the resulting CNN from *emotion_recognition* (accuracy: 61%)
* *real_time-g&e_recognition* - in this .py file *gender_vgg16model.h5* and *my_emotion_model.h5* are used for real-time recognition using webcam.

## How to run
Download the haar cascade classifier from the link above. You will have to import this algorithm in *real_time-g&e_recognition*.

You can directly download the two CNNs (*gender_vgg16model.h5* and *my_emotion_model.h5*), import them into the *real_time-g&e_recognition* and run it for real time recognition.
Alternatively you can train your CNNs using *gender_recognition* and *emotion_recognition* files and use them in *real_time-g&e_recognition*.

## License
[GNU GPLv2](https://choosealicense.com/licenses/gpl-2.0/)