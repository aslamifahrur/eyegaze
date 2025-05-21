import cv2
import numpy as np

# Buka kamera (0 = default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize agar lebih ringan (opsional)
    frame = cv2.resize(frame, (640, 480))

    # Fokuskan pada ROI (area mata) - sesuaikan nilai crop sesuai posisi kamera
    roi = frame[10:500, 30:790]  # [y1:y2, x1:x2]

    # Ubah ke grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Blur untuk mengurangi noise
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Threshold invers untuk mendeteksi pupil (biasanya hitam)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

    # Cari kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Pilih kontur terbesar sebagai kandidat pupil
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
        if radius > 2:  # filter noise kecil
            center = (int(x), int(y))
            cv2.circle(roi, center, int(radius), (0, 255, 0), 2)
            cv2.circle(roi, center, 2, (0, 0, 255), -1)
            # Tampilkan koordinat di layar
            cv2.putText(roi, f"Pupil: {center}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Tampilkan hasil
    cv2.imshow("ROI (Eye)", roi)
   # cv2.imshow("Threshold", thresh)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
