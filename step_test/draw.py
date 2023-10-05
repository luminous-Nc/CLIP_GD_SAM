from PIL import Image
import cv2
import numpy as np

if __name__ == "__main__":
    original_image = cv2.imread("img.png")
    # original_image = annotated_image
    result_image = original_image.copy()

    bb_xyxy = [100, 50, 300, 200]
    color = [0, 255, 0]
    cv2.rectangle(result_image, (bb_xyxy[0], bb_xyxy[1], bb_xyxy[2], bb_xyxy[3]), color, 2)
    text_position = (bb_xyxy[0], bb_xyxy[1] - 8)

    text_size, _ = cv2.getTextSize("haha", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_width, text_height = text_size
    text_width *= 2
    text_height *= 1
    bg_rect_start = (bb_xyxy[0], bb_xyxy[1])
    bg_rect_end = (text_position[0] + text_width, text_position[1] - text_height)

    # 绘制背景矩形
    cv2.rectangle(result_image, bg_rect_start, bg_rect_end, color, thickness=cv2.FILLED)

    cv2.putText(result_image, "haha", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("haha", result_image)
    cv2.waitKey(0)
