import cv2
import mediapipe as mp 

class UseHaarCascade:
    pass 

class UseMediaPipe:
    
    def __init__(self):
        pass 
    
    def _runMediaPipe(self):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()

        cap = cv2.VideoCapture(0)
        
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
                    
                    # Nose Landmark
                    nose = face_landmarks.landmark[1]
                    x = int(nose.x * w)
                    y = int(nose.y * h)
                    
                    # Draw Nose Landmark
                    cv2.circle(frame, (x,y), 5, (0,255,0), -1)
                    
                    center_x = w // 2
                    
                    # Attention Logic
                    if abs(x - center_x) < 80:
                        status = "F"
                        color = (0,255,0)
                    else:
                        status = "NF"
                        color = (0,0,255)
                    
                    # Draw Text
                    cv2.putText(frame, status, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
            cv2.imshow("Attention Monitoring", frame)
            
            if cv2.waitKey(1) == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
            
                        

class Main: 
    def __init__(self):
        self.method = "HARR_CASCADE"
    
    def _run(self, method):
        self.method = method
        
        if self.method == "HARR_CASCADE":
            pass 
        
        elif self.method == "MEDIA_PIPE":
           mediapipeobj = UseMediaPipe()
           mediapipeobj._runMediaPipe()
    