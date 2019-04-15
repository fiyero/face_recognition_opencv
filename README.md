#Build Face recognizer with OpenCV
## https://medium.com/@patrickhk/face-recognition-with-opencv-without-deep-learning-435cb6d36a53
![1](https://cdn-images-1.medium.com/max/800/1*mPSyO1IuFk19ULH8pzMNsg.jpeg)<br/>
#### In short, face recognition is doing the following things:

1. Detect and capture face present in the incoming sources, such as image file, webcam, IP cam or video file. We can achieve it by using Harr Cascade, Histogram of Oriented Gradients (HOG) or others.
2. We can consider identify the facial landmarks and apply some transformation to make the captured face image part fits better in our face recognizer
3. Encode the image information into vector therefore we can apply different Machine learning algo to do classification to predict which person is present in the incoming image sources.
4. Visualize the prediction by drawing a box with prediction on the image sources.
First we should prepare the training set, I just google and download some images for each of the 7 celebrities. I create their own folder and put images for easier later processing. Don’t have to care about the file names because they are not important(file extension is important).<br/>
![2](https://cdn-images-1.medium.com/max/800/1*3SA6Yrck_17rtdXqp2RByA.png)

#### You should have several things in your minds:<br/>
1. What is your image source? Webcam? IPcam? image file ? Video?
2. Should we pre-process the image? like resize/ turn to different color scale?
3. Image file is pixel, how to encode them?
4. How to extract facial part out of the whole image?
5. What ML algo are you going to apply? How to map output logit to label in text form?
6. How to export the prediction?<br/>

I have created different face recognizers by using OpenCV , dlib(by Davis King) and face_recognition(by Ageitgey). Using face_recognition to build face recognizer is very efficient and easy but I strongly recommend you to play with the sources code and try to rebuild in OpenCV and dlib. Otherwise you won’t really learn somethings. Do check out their tutorials, well explained. <br/>

## Lets start with the most easiest and basic one, by OpenCV and Harr Cascade
A common way to locate the facial part information is by using pre-trained CascadeClassifier. We can download the haarcascade_frontalface_default.xml and use it to locate the 4 coordinates of the facial part. To detect the Haar like features, gray scale image does better job than RGB image. Therefore we should convert the input image into gray scale. I suggest using PIL to convert into gray scale and numpy array.

With the 4 coordinates of the facial part, we can extract the facial part out of the whole image. With this information we can create our training data. Training the cv2.face.LBPHFaceRecognizer is super fast, maybe 1–2 seconds. Therefore you can expect the accuracy won’t be very high compared to deep learning model. We will draw a box to highlight the “face” on the image. You can also apply more cascades, such as eye and smile cascade to further detect more biological features. But for face recognition, face cascade classifier is enough.

Remember to build dict to map the integer label form to text form for output prediction.

By using the cv2.face.LBPHFaceRecognizer you will get two results, the label_id and confidence value. Worth notice that confidence value is the distance to the predicted label, therefore this value should be lower is better. Initially I didn’t realize it therefore the recognizer cannot predict well. You can set a threshold value for the confidence to reduce false positive result.

Here are some results:
![3](https://cdn-images-1.medium.com/max/600/1*KPoY58i7GRqTTG7eOI0RVA.jpeg)
![4](https://cdn-images-1.medium.com/max/600/1*CAbNl3JGmt8wbZQXQagRPQ.jpeg)

If you use webcam as your image source, you better get a higher quality one. Mine is quite bad…The quality of test image feeding to the recognizer makes a huge difference and directly affect the result. If I switch to feeding image file into the recognizer, the accuracy bumps up.

![5](https://cdn-images-1.medium.com/max/800/1*fFHIppS-Nl3OOntgMNKYtw.jpeg)
![6](https://cdn-images-1.medium.com/max/600/1*kJIj0M2vbyj-GlztHRs_nA.jpeg)

#### How to improve the accuracy of this face recognizer?
1. Use more training image per class. Actually I just use 8 images per class, which is definitely not enough. Just for experimental purpose.
2. Notice if there is imbalanced training sample. Although I put 8 images each class to train, some image may have multiple/ don’t have face feature detected by the pre-trained cascadeClassifier. Therefore the actual number of training array feeded per class is not necessarily equal to 8. Since the base number is low (8), missing/extra 1–2 training sample will contribute huge difference.
3. Garbage in garbage out, use higher quality image as training set will lead to better result
4. Fine tune the confidence threshold from the cv2.face.LBPHFaceRecognizer
5. If you resize the image for training set, you should apply the same augmentation to your test set
