from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import requests
import numpy as np
import cv2

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

DEVICE_STRING = "mps"
DEVICE_STRING = "cpu"

def pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image)
    return open_cv_image[:, :, ::-1].copy()


class Detection:
    def __init__(self) -> None:
        self.model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny").to(device=DEVICE_STRING)
        print(self.model.__class__)
        self.image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    def pred(self, image):
        cv2_image = image
        # print(image)
        image = Image.fromarray(image)

        inputs = self.image_processor(images=image, return_tensors="pt").to(device=DEVICE_STRING)
        print(cv2_image.shape)
        print(inputs['pixel_values'].shape)
        outputs = self.model(**inputs)

        # print results
        target_sizes = torch.tensor([image.size[::-1]]).to(device=DEVICE_STRING)
        results = self.image_processor.post_process_object_detection(
            outputs, threshold=0.8, target_sizes=target_sizes
        )[0]

        # a_img = pil_to_cv2(image)
        a_img = cv2_image
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))
            cv2.rectangle(a_img, start_point, end_point, (0, 0, 255), thickness=1, lineType=cv2.LINE_8)
            cv2.putText(a_img,
                        f"{self.model.config.id2label[label.item()]} with {round(score.item(), 3)}",
                        # f"cutie with 100%",
                        start_point,
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1,
                        color=(0, 255, 0),
                        thickness=2,
                        )
            # print(box)
            # box = [round(i, 2) for i in box.tolist()]
            # print(
            #     f"Detected {self.model.config.id2label[label.item()]} with confidence "
            #     f"{round(score.item(), 3)} at location {box}"
            # )
        return a_img


def main():
    device = DEVICE_STRING if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    a = Detection()
    # a.pred(pil_to_cv2(image))
    cam_feed = cv2.VideoCapture(1)
    cam_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        ret, img = cam_feed.read()
        if img is not None:
            annotated_image = a.pred(img)
            cv2.imshow("", annotated_image)

        if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1) == 27):
            break

    cam_feed.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
