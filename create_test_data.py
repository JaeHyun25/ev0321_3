import json
import os
import random
from collections import Counter

def create_test_data(image_dir: str, json_dir: str, output_path: str, num_samples: int = 100):
    """테스트 데이터 생성"""
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 랜덤하게 샘플 선택
    selected_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    test_data = []
    
    for image_file in selected_files:
        # 이미지 경로
        image_path = os.path.join(image_dir, image_file)
        
        # JSON 파일 경로
        json_file = image_file.rsplit('.', 1)[0] + '.json'
        json_path = os.path.join(json_dir, json_file)
        
        if not os.path.exists(json_path):
            continue
            
        # JSON 파일 읽기
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not data or len(data) == 0:
            continue
            
        # 첫 번째 번호판 정보 사용
        plate_info = data[0]
        
        # 테스트 데이터 항목 생성
        test_case = {
            'image_path': image_path,
            'plate_info': plate_info,
            'is_ev': plate_info.get('attrs', {}).get('ev', False)
        }
        
        test_data.append(test_case)
    
    # 데이터셋 통계 출력
    ev_count = sum(1 for item in test_data if item['is_ev'])
    non_ev_count = len(test_data) - ev_count
    
    print("\n=== 데이터셋 통계 ===")
    print(f"전체 샘플 수: {len(test_data)}")
    print(f"전기차 수: {ev_count} ({ev_count/len(test_data)*100:.2f}%)")
    print(f"일반차 수: {non_ev_count} ({non_ev_count/len(test_data)*100:.2f}%)")
    
    # 테스트 데이터 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n테스트 데이터 생성 완료: {len(test_data)}개 샘플")

if __name__ == "__main__":
    # 디렉토리 경로 설정
    image_dir = "/home/ijh/combined/CENTRALCITY_EVMONITORING/MISRECOG/"
    json_dir = "/home/ijh/combined/CENTRALCITY_EVMONITORING/MISRECOG_JSON/"
    output_path = "test_data.json"
    
    # 테스트 데이터 생성
    create_test_data(image_dir, json_dir, output_path) 