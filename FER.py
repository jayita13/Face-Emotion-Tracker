import cv2
from fer import FER
import warnings
warnings.filterwarnings("ignore")

detector = FER() 

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, image = cap.read()

    emotion = detector.top_emotion(image)
    
    cv2.putText(image,emotion,(10, image.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
    #cv2.putText(image,str(score),(10, image.shape[0] - 20),
                   # cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
    
    cv2.imshow('img',image)
    #print(result)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()
