import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


glasses = cv2.imread("filters/glasses.png", -1)
hat = cv2.imread("filters/hat.png", -1)
mustache = cv2.imread("filters/mustache.png", -1)


missing_filters = set()

def overlay_filter(frame, filter_img, x, y, w, h, position="center"):
    if filter_img is None:
        if position not in missing_filters:
            print(f"Warning: filter_img is None for {position}")
            missing_filters.add(position)
        return frame

    try:
        filter_resized = cv2.resize(filter_img, (w, h))
    except Exception as e:
        print(f"Resize failed for {position}: {e}")
        return frame

    y1, y2 = y, y + filter_resized.shape[0]
    x1, x2 = x, x + filter_resized.shape[1]

    if y1 < 0 or y2 > frame.shape[0] or x1 < 0 or x2 > frame.shape[1]:
        return frame

    if len(filter_resized.shape) == 3 and filter_resized.shape[2] == 4:
        alpha_filter = filter_resized[:, :, 3] / 255.0
        alpha_frame = 1.0 - alpha_filter
        for c in range(3):
            frame[y1:y2, x1:x2, c] = (
                alpha_filter * filter_resized[:, :, c] +
                alpha_frame * frame[y1:y2, x1:x2, c]
            )
    else:
        if len(filter_resized.shape) == 2:
            filter_resized_colored = cv2.cvtColor(filter_resized, cv2.COLOR_GRAY2BGR)
        elif len(filter_resized.shape) == 3 and filter_resized.shape[2] == 4:
            filter_resized_colored = filter_resized[:, :, :3]
        else:
            filter_resized_colored = filter_resized
        frame[y1:y2, x1:x2] = filter_resized_colored

    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
       
        frame = overlay_filter(frame, glasses, x, y + int(h / 4), w, int(h / 3), "glasses")

      
        hat_w = int(w * 2.0)
        hat_h = int(h * 1.2)
        hat_x = x - int((hat_w - w) / 2)
        hat_y = y - hat_h + int(h * 0.1)
        frame = overlay_filter(frame, hat, hat_x, hat_y, hat_w, hat_h, "hat")

     
        mustache_x = x + w // 4
        mustache_y = y + (2 * h) // 3
        mustache_w = w // 2
        mustache_h = h // 5
        frame = overlay_filter(frame, mustache, mustache_x, mustache_y, mustache_w, mustache_h, "mustache")

    cv2.imshow("Snapchat Filters", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
