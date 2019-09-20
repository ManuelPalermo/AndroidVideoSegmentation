# Android Video Segmentation
Project done in order to learn how to use DeepLearning models in mobile applications. A camera application with real time semantic segmentation was implemented using OpenCV(for image preprocessing and camera handling) and Google´s trained Deeplab3+ model(for video frame segmentation).

<br></br>
## Demos
<span>
    <img src="Demos/person.gif" alt="person" width="240" height="426">
    <img src="Demos/walking.gif" alt="walking" width="240" height="426">
    <img src="Demos/plants.gif" alt="plants" width="240" height="426">
    <img src="Demos/dog.gif" alt="dog" width="240" height="426">
    <img src="Demos/sofa.gif" alt="sofa" width="240" height="426">
    <img src="Demos/table_chairs.gif" alt="table_chairs" width="240" height="426">
</span>





### Helpful Resources:
* Use of quantized version of deeplab3+(for better inference speed): https://github.com/tantara/JejuNet
* Visualize .tflite model structure/inputs/outputs: https://github.com/lutzroeder/netron
* Fix opencv camera: https://heartbeat.fritz.ai/working-with-the-opencv-camera-for-android-rotating-orienting-and-scaling-c7006c3e1916
