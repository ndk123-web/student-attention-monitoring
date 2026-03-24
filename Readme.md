## What will Be Required ?


### Language 
- Python

### OpenCV library
- open source machine vision library
- feature:
    - open the camera
    - read image
    - process image
    - draw shape
    - detect face

### Detect A Face / Multiple Faces
1. Haar Cascade (Opencv built in library) (Old)
    - Answers: whether face or not ?
    - Pre Train Model which tells is it face or not 
    - Steps:
        - it has fixed size window assume (24*24) 
        - it checks for each position of 24*24 and checks whether it is face or not on every shrink
        - it uses sliding window (top-bottom and left-right check in each window)

2. MediaPipe (Made by Google can detect 467 meshes)
    - Answers: exact structure of face (468 face landmarks)
