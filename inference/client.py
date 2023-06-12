from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
import requests
import cv2
import os
import io

DEVICE_STRING = "cpu"
URL = "http://localhost:8080/"


def pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image)
    return open_cv_image[:, :, ::-1].copy()


def main():

    os.environ['YOLO_MODE'] = 'CLIENT'
    client_model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny").to(device=DEVICE_STRING)

    os.environ['YOLO_MODE'] = 'NONE'
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = image_processor(images=image, return_tensors="pt").to(device=DEVICE_STRING)

    pixel_values = inputs['pixel_values']
    p_height=pixel_values.shape[-2]
    p_width=pixel_values.shape[-1]
    print(p_height, p_width)

    outputs = client_model(**inputs)
    # outputs = torch.tensor([1.0, 1.0, 1.0])

    server_inputs = outputs
    buff = io.BytesIO()
    torch.save(server_inputs, buff)
    buff.seek(0)

    r = requests.post(url=URL, data=buff, verify=False, headers={'Content-Type': 'application/octet-stream', 'Content-Length': f"{buff.getbuffer().nbytes}"})

    input_buff = io.BytesIO()
    input_buff.write(r.content)
    input_buff.seek(0)

    outputs = torch.load(input_buff, map_location=DEVICE_STRING)
    target_sizes = torch.tensor([image.size[::-1]]).to(device=DEVICE_STRING)
    results = image_processor.post_process_object_detection(
        outputs, threshold=0.8, target_sizes=target_sizes
    )[0]

    a_img = pil_to_cv2(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        cv2.rectangle(a_img, start_point, end_point, (0, 0, 255), thickness=1, lineType=cv2.LINE_8)
        cv2.putText(a_img,
                    f"{client_model.config.id2label[label.item()]} with {round(score.item(), 3)}",
                    start_point,
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1,
                    color=(0, 255, 0),
                    thickness=2,
                    )

    img2 = a_img[:,:,::-1]
    plt.imshow(img2)
    plt.show()


if __name__ == '__main__':
    main()
