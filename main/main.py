import cv2
import mediapipe as mp 
import math

class UseHaarCascade:
    pass 

    """
        UseMediaPipe class implements face detection using MediaPipe's Face Mesh solution. It captures video from the default camera, processes each frame to detect facial landmarks, and specifically tracks the nose landmark to determine if the user is looking at the center of the screen. The detected nose position is used to display a status ("F" for facing forward and "NF" for not facing forward) on the video feed. The video capture continues until the user presses the ESC key.
    """
class UseMediaPipe:
    
    def __init__(self):
        pass 
    
    def _runMediaPipe(self):
        if not hasattr(mp, "solutions"):
            print("Your current mediapipe build does not expose 'solutions'.")
            print("Install a legacy-compatible version, e.g.: pip install mediapipe==0.10.14")
            return

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Could not open camera device")
            return
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to capture video")
                break
                
            
            h,w,_ = frame.shape
            
            # BGR means Blue, Green, Red. OpenCV uses BGR format by default, so we need to convert the captured frame from BGR to RGB format before processing it with MediaPipe. This is done using cv2.cvtColor() function, which takes the input frame and the color conversion code cv2.COLOR_BGR2RGB as arguments. The resulting 'rgb' variable will contain the frame in RGB format, which is suitable for processing with MediaPipe's face mesh detection.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            
            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    
                    # Get all x and y coordinates
                    xs = [int(lm.x * w) for lm in face_landmarks.landmark]
                    ys = [int(lm.y * h) for lm in face_landmarks.landmark]

                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)

                    # Face center
                    face_center_x = (xmin + xmax) // 2
                    face_center_y = (ymin + ymax) // 2

                    # Nose
                    x = int(face_landmarks.landmark[1].x * w)
                    y = int(face_landmarks.landmark[1].y * h)

                    face_width = xmax - xmin
                    face_height = ymax - ymin
                    
                    distance = math.sqrt((x - face_center_x)**2 + (y - face_center_y)**2)
                    threashold = 0.25 * face_width

                    if distance < threashold:
                        status = "F"
                        color = (0,255,0)
                    else:
                        status = "NF"
                        color = (0,0,255)

                    # Draw box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

                    # Draw nose
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                    # Text
                    cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
            cv2.imshow("Attention Monitoring", frame)
            
            if cv2.waitKey(1) == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
            
                        

class Main: 
    def __init__(self):
        self.method = "HARR_CASCADE"
    
        """it runs the Main Function
        """
    def _run(self, method):
        self.method = method
        
        if self.method == "HARR_CASCADE":
            pass 
        
        elif self.method == "MEDIA_PIPE":
           mediapipeobj = UseMediaPipe()
           mediapipeobj._runMediaPipe()


if __name__ == "__main__":
    app = Main()
    app._run("MEDIA_PIPE")
    