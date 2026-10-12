# Assignment 9: Object (person) detection from CCTV footage
import cv2

# vtest.avi is a CCTV style street video from OpenCV samples
cap = cv2.VideoCapture("vtest.avi")
print("Total frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))

# HOG + SVM pre-trained people detector (comes with OpenCV)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# background subtractor to find moving objects
bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40)

out = cv2.VideoWriter("output_detected.avi", cv2.VideoWriter_fourcc(*'XVID'), 10, (768, 576))
frame_no = 0
saved = [50, 150, 250]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1

    # moving object mask
    mask = bg.apply(frame)

    # detect people every frame
    # people are small in CCTV, so enlarge frame 2x before detection
    big = cv2.resize(frame, None, fx=2, fy=2)
    boxes, weights = hog.detectMultiScale(big, winStride=(8, 8), padding=(8, 8), scale=1.05)
    boxes = [[int(v / 2) for v in b] for b in boxes]     # scale boxes back
    weights = [float(w) for w in weights]

    # remove very big boxes (a person in this camera view is not taller than ~200 px)
    filtered = [(b, w) for b, w in zip(boxes, weights) if b[3] < 200]
    boxes = [f[0] for f in filtered]
    weights = [f[1] for f in filtered]

    # non max suppression to remove overlapping boxes
    keep = cv2.dnn.NMSBoxes(boxes, weights, 0.5, 0.3)
    count = 0
    for i in keep:
        (x, y, w, h), conf = boxes[i], weights[i]
        count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"person {conf:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    cv2.putText(frame, f"Frame {frame_no}  People: {count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    out.write(frame)

    if frame_no in saved:
        cv2.imwrite(f"a9_frame{frame_no}.png", frame)
        cv2.imwrite(f"a9_mask{frame_no}.png", mask)
        print(f"Frame {frame_no}: {count} people detected")

    if frame_no % 50 == 0:
        print(f"Processed {frame_no} frames...")

    if frame_no >= 300:   # limit for lab demo
        break

cap.release()
out.release()
print("Detection done. Output saved as output_detected.avi")
