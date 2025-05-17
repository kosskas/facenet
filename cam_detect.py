import cv2
import numpy as np
from tensorflow.keras.models import load_model


def scaling(x, scale):
    return x * scale


model = load_model("model2.h5", custom_objects={"scaling": scaling})


def preprocess_digit(roi):
    # Konwersja do odcieni szarości
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Zmiana rozmiaru
    roi = cv2.resize(roi, (160, 160))

    # Odwrócenie kolorów jeśli tło jest ciemne
    mean_val = np.mean(roi)
    if mean_val < 127:
        roi = cv2.bitwise_not(roi)

    # Konwersja z powrotem do 3 kanałów (dla modelu)
    roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

    # Normalizacja i przygotowanie wejścia
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=0)  # (1, 160, 160, 3)

    return roi


def detect_white_paper(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 10000:
            x, y, w, h = cv2.boundingRect(largest_contour)
            roi = frame[y : y + h, x : x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            digit_img = preprocess_digit(roi)
            prediction = model.predict(digit_img)
            digit = np.argmax(prediction)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                str(digit),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                2,
            )

    return frame


# Główna pętla
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_white_paper(frame)

    cv2.imshow("Digit Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
