import gradio as gr
from ultralytics import YOLO
import cv2
import pandas as pd

# Load model
model = YOLO("grocery_yolo_best.pt")

# Product prices
price_map = {
    "Arla-Standard-Milk": 60,
    "Bravo-Orange-Juice": 95,
    "God-Morgon-Apple-Juice": 100,
    "Tropicana-Apple-Juice": 110,
    "Valio-Vanilla-Yoghurt": 45
}

def detect(image):

    results = model(image, conf=0.5)[0]

    cart = {}
    total = 0

    for box in results.boxes:

        cls_id = int(box.cls)
        label = model.names[cls_id]
        price = price_map.get(label, 0)

        # Count quantity
        if label not in cart:
            cart[label] = {"price": price, "qty": 1}
        else:
            cart[label]["qty"] += 1

        total += price

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(image,f"{label} ₹{price}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)

    # Create billing table
    rows = []
    for item,data in cart.items():
        rows.append([item, data["qty"], data["price"], data["qty"]*data["price"]])

    df = pd.DataFrame(rows, columns=["Product","Quantity","Price","Total"])

    return image, df, total


interface = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="numpy", label="Upload Grocery Image"),
    outputs=[
        gr.Image(label="Detected Products"),
        gr.Dataframe(label="Billing Table"),
        gr.Number(label="Total Bill ₹")
    ],
    title="Smart Grocery Billing System (AI Powered)",
    description="Upload a grocery image to detect products and automatically generate the bill."
)

interface.launch()