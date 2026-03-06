
import cv2
import numpy as np
import tensorflow as tf
import time

# Load trained classification model
model = tf.keras.models.load_model("grocery_5_products_model.h5")

class_names = [
    "Bravo-Orange-Juice",
    "Milk",
    "Soap",
    "Tropicana-Apple-Juice",
    "Yogurt"
]

price_map = {
    "Tropicana-Apple-Juice": 110,
    "Bravo-Orange-Juice": 95,
    "Milk": 60,
    "Yogurt": 45,
    "Soap": 35
}

cart = {}
last_scan_time = 0
scan_delay = 2  # seconds between scans

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Define scan area (center box)
    x1, y1 = w//3, h//3
    x2, y2 = 2*w//3, 2*h//3

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(frame, "Place item here",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2)

    scan_region = frame[y1:y2, x1:x2]

    img = cv2.resize(scan_region, (224,224))
    img = np.expand_dims(img, axis=0) / 255.0

    predictions = model.predict(img, verbose=0)
    class_id = np.argmax(predictions)
    label = class_names[class_id]
    confidence = predictions[0][class_id]

    current_time = time.time()

    if confidence > 0.90 and (current_time - last_scan_time > scan_delay):
        cart[label] = cart.get(label, 0) + 1
        last_scan_time = current_time

    # Display cart
    y_offset = 30
    total = 0

    for item, qty in cart.items():
        price = price_map.get(item, 0)
        total += price * qty
        cv2.putText(frame,
                    f"{item} x{qty} = ₹{price*qty}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,0,0),
                    2)
        y_offset += 20

    cv2.putText(frame,
                f"Total: ₹{total}",
                (10, y_offset+20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255),
                3)

    cv2.imshow("Smart Grocery Billing (No YOLO)", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()