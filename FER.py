import cv2
from fer import FER
from imutils.video import VideoStream

detector = FER() 
output = "emotion.avi"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
cap = VideoStream(src=0).start()
while True:
    frame = cap.read()
    (H, W) = frame.shape[:2]
    result = detector.top_emotion(frame)
    
    cv2.putText(frame,str(result[0]),(10, H - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
    
    out = cv2.VideoWriter(output, fourcc, 20.0, (W, H))
    out.write(frame)
    cv2.imshow('img',frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()
out.release()
