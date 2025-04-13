from ultralytics import YOLO
import time
import cv2
import settings

def load_model(model_path):
    return YOLO(model_path)

def classify_waste_type(detected_items):
    recyclable_items = set(detected_items) & set(settings.RECYCLABLE)
    non_recyclable_items = set(detected_items) & set(settings.NON_RECYCLABLE)
    hazardous_items = set(detected_items) & set(settings.HAZARDOUS)
    return recyclable_items, non_recyclable_items, hazardous_items

def remove_dash_from_class_name(class_name):
    return class_name.replace("_", " ")

def _display_detected_frames(model, image):
    image = cv2.resize(image, (1080, int(1080 * (9 / 16))))
    res = model.predict(image, conf=0.6)
    names = model.names
    detected_items = set()

    for result in res:
        detected_items.update([names[int(c)] for c in result.boxes.cls])

    recyclable_items, non_recyclable_items, hazardous_items = classify_waste_type(detected_items)

    # Prepare text for display
    text_lines = []
    if recyclable_items:
        text_lines.extend([f"{remove_dash_from_class_name(item)} - Recyclable" for item in recyclable_items])
    if non_recyclable_items:
        text_lines.extend([f"{remove_dash_from_class_name(item)} - Non-Recyclable" for item in non_recyclable_items])
    if hazardous_items:
        text_lines.extend([f"{remove_dash_from_class_name(item)} - Hazardous" for item in hazardous_items])

    # Annotate the frame with detected items and waste types
    y_offset = 20
    for line in text_lines:
        cv2.putText(image, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        y_offset += 20

    result_image = res[0].plot()
    combined_image = cv2.addWeighted(result_image, 0.7, image, 0.3, 0)
    cv2.imshow("Garbage Segregation", combined_image)

def play_webcam(model):
    vid_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while vid_cap.isOpened():
        success, image = vid_cap.read()
        if success:
            _display_detected_frames(model, image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return
        else:
            break

    vid_cap.release()
    cv2.destroyAllWindows()
