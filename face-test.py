import cv2 

class HaarCascade:
    """
    Haar Cascade-based face detection implementation using OpenCV.
    This class initializes a Haar Cascade classifier for frontal face detection and captures video from the default camera. It processes each video frame to detect faces and draws rectangles around detected faces before displaying the video feed. The video capture continues until the user presses the ESC key.
    """
    def __init__(self):
        self.name = "Ndk"
    
    def _detect_faces(self):
        # Load the Haar Cascade XML file for frontal face detection using cv2.CascadeClassifier. The XML file should be in the same directory as the script or provide the correct path to it.
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
        if (face_cascade.empty()):
            print("Failed to load Haar Cascade XML file")
            return
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret , frame = cap.read()
            
            if not ret:
                print("Failed to capture video")
                break
            
            # Important: Convert the captured frame to grayscale using cv2.cvtColor() before applying the face detection algorithm. Haar Cascade works better on grayscale images, and this step is crucial for accurate face detection.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # scaleFactor: This parameter specifies how much the image size is reduced at each image scale. A value of 1.1 means that the image will be reduced by 10% at each scale. This helps in detecting faces of different sizes in the image.
            # minNeighbors: This parameter specifies how many neighbors each candidate rectangle should have to retain it. A higher value means that only rectangles with more neighboring rectangles will be retained, which can help reduce false positives.
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
            
            
            for (x,y,w,h) in faces: 
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            
            cv2.imshow("Camera", frame)
            
            if cv2.waitKey(1) == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()

obj1 = HaarCascade()
obj1._detect_faces()