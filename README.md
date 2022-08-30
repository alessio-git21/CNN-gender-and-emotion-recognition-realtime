# Real time gender and emotion recognition using CNNs.
The purpose of this project is to train Convolutional Neural Networks (CNNs) to 
recognize the gender and the emotion of a person from his face. The trained CNNs are used for real-time recognition using computer's webcam.


https://user-images.githubusercontent.com/100300894/184353856-1b598501-12f4-47a4-b5de-dfc459564d79.mp4


## A brief description
Two CNNs are trained: one CNN for gender recognition ([dataset for gender recognition](https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset)) 
and the other one for emotion recognition ([dataset for emotion recognition](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)). 
The datasets are downloaded using the kaggle API.

The CNN for gender recognition is based on the pre-trained VGG16 model.

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
* *gender_recognition* - a IPYNB file where the CNN for gender recognition is trained (actual accuracy: 95%)
* *emotion_recognition* - a IPYNB file where the CNN for emotion recognition is trained (actual accuracy: 61%)
* *real_time-g&e_recognition* - in this .py file the CNNs are used for real-time recognition using webcam.

## How to run
Download the haar cascade classifier from the link above. You will have to import this algorithm in *real_time-g&e_recognition*.

Train your own CNNs using *gender_recognition* and *emotion_recognition* files and save the resulting models in .h5 format. Load the models in *real_time-g&e_recognition* file and run it. Once you have launched this file:
* press G for gender recognition
* press E for emotion recognition
* press Q to quit.

During training you are encouraged to modify data preprocessing (for instance you can use data augumentation) and the models parameters to improve the performance.

## License
[GNU GPLv2](https://choosealicense.com/licenses/gpl-2.0/)
