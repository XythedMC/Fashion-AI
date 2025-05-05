import cv2
import numpy as np
from ultralytics import YOLO


import webcolors

def predict(frame, model):
    results = model(frame)
    return results

def get_color_name(hex_color):
    try:
        # Get the closest color name
        color_name = webcolors.hex_to_name(hex_color)
    except ValueError:
        # If exact match not found, find the closest color
        closest_name = None
        min_distance = float('inf')
        for name, hex_value in webcolors._definitions._CSS3_NAMES_TO_HEX.items():
            r, g, b = webcolors.hex_to_rgb(hex_value)
            input_r, input_g, input_b = webcolors.hex_to_rgb(hex_color)
            distance = sum((c1 - c2) ** 2 for c1, c2 in zip((input_r, input_g, input_b), (r, g, b)))
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        color_name = closest_name
    return color_name

def plot_bboxes(results, frame, class_names):
    xyxys = []
    confidences = []
    class_ids = []
    average_colors = []
    color_names = []

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            xyxy = box.xyxy[0]  # Extract bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = box.cls[0]    # Class ID

            x1, y1, x2, y2 = map(int, xyxy)

            center_x, center_y = (x1 + x2) // 2 + (x2 - x1) // 4, (y1 + y2) // 2  # Shift by +70 on the x-axis
            small_box_x1 = center_x - 5
            small_box_y1 = center_y - 5
            small_box_x2 = center_x + 5
            small_box_y2 = center_y + 5
            print(f"Bounding box size: {x2 - x1} x {y2 - y1}")

            # Append bounding box data
            xyxys.append(xyxy)
            confidences.append(conf)
            class_ids.append(cls)

            # Extract the region of interest (ROI) for the bounding box
            
            roi = frame[small_box_y1:small_box_y2+1, small_box_x1:small_box_x2+1]

            # Calculate the average color of the ROI
            if roi.size > 0:  # Ensure the ROI is not empty
                avg_color = np.mean(roi, axis=(0, 1))  # Average over height and width
                avg_color_rgb = avg_color[::-1]  # Convert BGR to RGB
                avg_color_hex = '#%02x%02x%02x' % tuple(map(int, avg_color_rgb))  # Convert to HEX
                average_colors.append(avg_color_hex)

                # Get the color name
                color_name = get_color_name(avg_color_hex)
                color_names.append(color_name)

            # Draw the bounding box on the frame
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            frame = cv2.rectangle(frame, (small_box_x1, small_box_y1), (small_box_x2, small_box_y2), (0, 0, 255), 2)
            frame = cv2.putText(frame, f"Class: {class_names[int(cls)]}, Conf: {conf:.2f}, Color: {color_name}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for color, name in zip(average_colors, color_names):
        print(f"Average color (HEX): {color}, Name: {name}")
    return frame

def saturate_frame(frame, saturation_scale=1.5):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Scale the saturatiosn channel
    h, s, v = cv2.split(hsv)
    s = np.clip(s * saturation_scale, 0, 255).astype(np.uint8)
    
    # Merge the channels back and convert to BGR
    hsv_saturated = cv2.merge([h, s, v])
    saturated_frame = cv2.cvtColor(hsv_saturated, cv2.COLOR_HSV2BGR)
    
    return saturated_frame

def add_padding(frame, padding=400):
    # Add padding to all sides of the frame
    padded_frame = cv2.copyMakeBorder(
        frame,
        top=padding,
        bottom=padding,
        left=padding,
        right=padding,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # Black padding
    )
    return padded_frame

def main():
    model = YOLO("runs/detect/train2/weights/best.pt")  # load a custom model
    class_names = ["t_shirt","sweater","short_sleeved_outwear","long_sleeved_outwear","vest","sling","shorts","trousers","skirt","short_sleeved_dress","long_sleeved_dress","vest_dress","sling_dress"]
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Cannot open camera"
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Resize the frame
        # frame = cv2.resize(frame, (640, 640))
        
        frame = add_padding(frame, padding=0)
        frame = cv2.flip(frame, 1)
        # Saturate the frame
        frame = saturate_frame(frame, saturation_scale=1.5)
        
        # Perform prediction
        results = predict(frame, model)
        
        # Plot bounding boxes
        frame = plot_bboxes(results, frame, class_names)
        
        # Display the frame
        cv2.imshow("Res", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
