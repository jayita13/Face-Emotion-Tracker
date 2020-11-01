import cv2
from fer import FER
from imutils.video import VideoStream

detector = FER() 

cap = VideoStream(src=0).start()
out = cv2.VideoWriter('emo.avi', cv2.VideoWriter_fourcc(*'MJPG'),20,(640,480)) 

while True:
    frame = cap.read()
    (H, W) = frame.shape[:2]
    result = detector.top_emotion(frame)
    
    cv2.putText(frame,str(result[0]),(200, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
 
    out.write(frame)
    cv2.imshow('img',frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()
out.release()
