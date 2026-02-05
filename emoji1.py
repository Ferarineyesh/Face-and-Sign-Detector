import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    expression = "😐"

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark

        mouth_width = dist(lm[61], lm[291])
        mouth_open = dist(lm[13], lm[14])

        brow_left = dist(lm[70], lm[159])
        brow_right = dist(lm[300], lm[386])

        # LOGIC
        if mouth_open > 0.05:
            expression = ":0"
        elif mouth_width > 0.08:
            expression = ":D"
        elif brow_left < 0.015 and brow_right < 0.015:
            expression = ">:v"
        elif brow_left > brow_right + 0.01:
            expression = ":l"
        elif mouth_width < 0.045:
            expression = ":("
        else:
            expression = ":)"

        x = int(lm[10].x * w)
        y = int(lm[10].y * h) - 80

        cv2.putText(
            frame,
            expression,
            (x - 40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (0, 255, 255),
            5,
            cv2.LINE_AA
        )

    cv2.imshow("Emoji Cam", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


