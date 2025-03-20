import cv2
import numpy as np
import json

def get_plate_info(image_path):
    """이미지에서 번호판 정보를 수동으로 추출하는 도우미 함수"""
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return None
    
    # 이미지 크기 출력
    height, width = image.shape[:2]
    print(f"이미지 크기: {width}x{height}")
    
    # 이미지 표시
    cv2.imshow("Image", image)
    print("\n번호판 영역을 선택하려면:")
    print("1. 마우스 왼쪽 버튼으로 드래그하여 영역 선택")
    print("2. Enter 키를 눌러 선택 완료")
    print("3. 'q' 키를 눌러 종료")
    
    # ROI 선택
    roi = cv2.selectROI("Image", image, False)
    cv2.destroyAllWindows()
    
    # 선택된 영역 정보
    x, y, w, h = roi
    print(f"\n선택된 영역:")
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"width: {w}")
    print(f"height: {h}")
    
    # 번호판 텍스트 입력 받기
    plate_text = input("\n번호판 텍스트를 입력하세요: ")
    
    # 번호판 정보 구성
    plate_info = {
        "area": {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "angle": 0  # 기본값
        },
        "text": plate_text
    }
    
    # JSON 파일로 저장
    output_file = "plate_info.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(plate_info, f, indent=4, ensure_ascii=False)
    
    print(f"\n번호판 정보가 {output_file}에 저장되었습니다.")
    return plate_info

if __name__ == "__main__":
    # 이미지 경로 입력 받기
    image_path = input("차량 이미지 경로를 입력하세요: ")
    plate_info = get_plate_info(image_path)
    
    if plate_info:
        print("\n생성된 번호판 정보:")
        print(json.dumps(plate_info, indent=4, ensure_ascii=False)) 