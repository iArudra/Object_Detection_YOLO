import cv2
import numpy as np

def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg.txt")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    with open("coco.names.txt") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes, output_layers
def detect_objects(img, net, output_layers):
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return outputs, height, width

def get_box_info(outputs, height, width, classes, conf_threshold=0.5, nms_threshold=0.4):
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return boxes, confidences, class_ids, indices

def draw_labels(frame, boxes, confidences, class_ids, indices, classes):
    for i in indices:
        if isinstance(i, list) or isinstance(i, tuple):  # If i is a list or tuple
            i = i[0]
        # Use i here
        label = classes[class_ids[i]]
        color = (0, 255, 0)  # Example color for bounding box
        cv2.rectangle(frame, (boxes[i][0], boxes[i][1]), (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]), color, 2)
        cv2.putText(frame, f"{label} {confidences[i]:.2f}", (boxes[i][0], boxes[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    net, classes, output_layers = load_yolo()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        outputs, height, width = detect_objects(frame, net, output_layers)
        boxes, confidences, class_ids, indices = get_box_info(outputs, height, width, classes)
        draw_labels(frame, boxes, confidences, class_ids, indices, classes)
        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

