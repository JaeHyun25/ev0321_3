import cv2
import numpy as np

# 테스트 이미지 생성 (1920x1080 크기)
image = np.ones((1080, 1920, 3), dtype=np.uint8) * 255

# 번호판 영역 그리기 (테스트용)
cv2.rectangle(image, (100, 100), (300, 150), (0, 0, 0), 2)

# 이미지 저장
cv2.imwrite("test_car.jpg", image)
print("테스트 이미지가 생성되었습니다: test_car.jpg") 