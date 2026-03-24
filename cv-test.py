'''
Copyright (c) 2024 Navnath Kadam. All rights reserved.
Author: Navnath Kadam
'''

'''
    cv2 come from the opencv-python library, which is a popular computer vision library in Python. It provides various functions for image processing and computer vision tasks. In this code snippet, we are using cv2 to read an image file named "test.png", display it in a window titled "MyImage", and wait for a key press before closing the window.
'''
import cv2

'''
    Learning about How Image Display works in OpenCV. I am creating a class named Image
'''
class OpenCvImage:
    def __init__(self):
        self.name = "Ndk"
    
    ''' 
        This method takes an image path as an argument, reads the image using cv2.imread(), and displays it in a window using cv2.imshow(). The window will remain open until a key is pressed, at which point it will be closed using cv2.destroyAllWindows().
    '''
    def display_image(self, image_path):
        
        # Read the image from the specified path (returns a NumPy array representing the image)
        img = cv2.imread(image_path) 
        
        # Display the image in a window titled "MyImage"
        cv2.imshow("MyImage", img)
        
        # Wait indefinitely for a key press (0 means wait until a key is pressed)
        cv2.waitKey(0)
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
# Create an instance of the OpenCvImage class and call the display_image method with the path to the image file
# obj1 = OpenCvImage()
# obj1.display_image("test.jpg")

class OpenCvVideoCapture:
    def __init__(self):
        self.name = "Ndk"
    
    def display_video(self):
        
        # Open the default camera (usually the built-in webcam) using cv2.VideoCapture(0). The argument '0' specifies the first camera device. If you have multiple cameras, you can use '1', '2', etc. to access them.
        cap = cv2.VideoCapture(0)

        # Check if the camera device was successfully opened. If not, print an error message and return from the function.
        if not cap.isOpened():
            print("Could not open camera device")
            return
        
        # Enter a loop to continuously read frames from the camera. The loop will run until the user presses the ESC key (ASCII code 27).
        while True:
            
            # Read a frame from the camera using cap.read(). This function returns two values: 'ret' (a boolean indicating whether the frame was successfully read) and 'frame' (the actual image frame captured from the camera).
            ret, frame = cap.read()
            
            # Check if the frame was successfully read. If 'ret' is False, it means there was an error capturing the video, and we print an error message and break out of the loop.
            if not ret:
                print("Failed to capture video")
                break
            
            # Display the captured frame in a window titled "Camera" using cv2.imshow(). This will show the live video feed from the camera.
            cv2.imshow("Camera", frame)
            
            # Wait for 1 millisecond for a key press using cv2.waitKey(1). If the user presses the ESC key (ASCII code 27), we break out of the loop and stop the video capture.
            if cv2.waitKey(1) == 27: # 27 is the ASCII code for the ESC key
                break
        
        # After exiting the loop, we release the camera resource using cap.release() and close all OpenCV windows using cv2.destroyAllWindows().
        cap.release()
        # Close all OpenCV windows
        cv2.destroyAllWindows()

obj2 = OpenCvVideoCapture()
obj2.display_video()
