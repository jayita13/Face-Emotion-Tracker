import cv2

from fer import FER

detector = FER(mtcnn=True) # or with mtcnn=False for Haar Cascade Classifier

image = cv2.imread("angry.jpg")
#cap = cv2.VideoCapture(0)
#while True:
    #_, image = cap.read()

result = detector.detect_emotions(image)

# Result is an array with all the bounding boxes detected.
bounding_box = result[0]["box"]
emotions = result[0]["emotions"]

cv2.rectangle(
        image,
        (bounding_box[0], bounding_box[1]),
        (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
        (0, 155, 255),
        2,
    )

for idx, (emotion, score) in enumerate(emotions.items()):
    color = (211, 211, 211) if score < 0.01 else (255, 0, 0)
    emotion_score = "{}: {}".format(
          emotion, "{:.2f}".format(score) if score > 0.01 else ""
        )
    cv2.putText(
            image,
            emotion_score,
            (bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + idx * 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
cv2.imwrite("emotion1.jpg", image)
print(result)
    
