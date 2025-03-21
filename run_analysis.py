from data_analysis_report import analyze_sensor_data

# CSV 파일 분석 실행
csv_path = '/home/ijh/auto_llm/LLMs/250109_21data.csv'
report_path = analyze_sensor_data(
    csv_path,
    chunk_size=5000,  # 메모리 사용량 최적화를 위해 작은 청크 크기 사용
    output_dir="analysis_results"
) 