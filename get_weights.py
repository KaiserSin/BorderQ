import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo11x.pt",
    confidence_threshold=0.05,
    device="mps",
    image_size=1280,
)

result = get_sliced_prediction(
    "resources/1 photo.png",
    model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.3,
    overlap_width_ratio=0.3,
    postprocess_type="NMS",
    postprocess_match_metric="IOS",
    postprocess_match_threshold=0.2,
)

people = [obj for obj in result.object_prediction_list if obj.category.id == 0]
print(f"Найдено людей: {len(people)}")

result.object_prediction_list = people

result.export_visuals(
    export_dir="resources/",
    file_name="result",
    hide_conf=True,
    hide_labels=True,
)
img = cv2.imread("resources/result.png")
cv2.imshow("YOLO11x + SAHI", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
