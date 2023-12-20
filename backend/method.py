import cv2

# draw the bouding box and save image
def draw_bbox(image_name, image, bboxes, color=(0, 255, 0), thickness=2):
    result_path = "../detection_result/"
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.imwrite(f"{result_path}detected_{image_name}", image)


# save images with bounding boxes and return result(json)
def predict_result(predictions, image_name):
    results = []
    for prediction in predictions:
        image = prediction.orig_img
        bboxes = prediction.boxes.xyxy
        draw_bbox(image_name, image, bboxes)
        bboxes_list = [[num.item() for num in bbox] for bbox in bboxes.cpu().numpy()]
        results.append({
                "filename": image_name,
                "class_name": list(prediction.names.items()),
                "predictions_bbox": bboxes_list
            })
    return results
