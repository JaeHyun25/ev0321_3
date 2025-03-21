import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import process_sensor_data
from datetime import datetime
import os

class SensorDataAnalyzer:
    def __init__(self, csv_path: str):
        """
        센서 데이터 분석기 초기화
        
        Args:
            csv_path: CSV 파일 경로
        """
        self.csv_path = csv_path
        self.raw_data = None
        self.processed_data = None
        self.report_path = None
        
    def load_data(self, chunk_size=10000):
        """
        데이터 로드 및 기본 전처리
        대용량 파일을 위해 청크 단위로 처리
        
        Args:
            chunk_size: 한 번에 처리할 행 수
        """
        try:
            # 파일 크기 확인
            file_size = os.path.getsize(self.csv_path) / (1024 * 1024)  # MB 단위
            print(f"파일 크기: {file_size:.2f} MB")
            
            # 청크 단위로 데이터 읽기
            chunks = []
            for chunk in pd.read_csv(self.csv_path, chunksize=chunk_size):
                chunks.append(chunk)
                print(f"청크 처리 중... ({len(chunks)}개 청크 완료)")
            
            # 청크들을 하나의 데이터프레임으로 합치기
            self.raw_data = pd.concat(chunks)
            
            # timestamp 컬럼 처리
            if 'timestamp' in self.raw_data.columns:
                self.raw_data['timestamp'] = pd.to_datetime(self.raw_data['timestamp'])
                self.raw_data.set_index('timestamp', inplace=True)
            
            print(f"데이터 로드 완료: {self.raw_data.shape[0]}행, {self.raw_data.shape[1]}열")
            return True
        except Exception as e:
            print(f"데이터 로드 실패: {str(e)}")
            return False
            
    def analyze_missing_values(self) -> dict:
        """결측치 분석"""
        missing_stats = {
            'total_columns': len(self.raw_data.columns),
            'columns_with_missing': (self.raw_data.isnull().sum() > 0).sum(),
            'missing_percentages': (self.raw_data.isnull().sum() / len(self.raw_data) * 100).to_dict()
        }
        return missing_stats
        
    def generate_plots(self, output_dir: str):
        """분석 그래프 생성"""
        # 메모리 사용량 최적화를 위해 데이터 샘플링
        sample_size = min(10000, len(self.raw_data))
        sampled_data = self.raw_data.sample(n=sample_size) if len(self.raw_data) > sample_size else self.raw_data
        sampled_processed = self.processed_data.loc[sampled_data.index] if self.processed_data is not None else None
        
        # 결측치 히트맵
        plt.figure(figsize=(12, 6))
        sns.heatmap(sampled_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
        plt.title('결측치 분포 히트맵 (샘플 데이터)')
        plt.savefig(f"{output_dir}/missing_values_heatmap.png")
        plt.close()
        
        # 처리 전/후 시계열 그래프 (처음 5개 컬럼에 대해)
        for col in sampled_data.columns[:5]:
            plt.figure(figsize=(12, 6))
            plt.plot(sampled_data.index, sampled_data[col], label='원본 데이터', alpha=0.5)
            if sampled_processed is not None:
                plt.plot(sampled_processed.index, sampled_processed[col], label='처리된 데이터', alpha=0.5)
            plt.title(f'{col} - 처리 전/후 비교 (샘플 데이터)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{col}_comparison.png")
            plt.close()
            
    def generate_report(self, output_dir: str = "analysis_results"):
        """분석 리포트 생성"""
        import os
        from datetime import datetime
        
        # 결과 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 데이터 처리
        print("데이터 처리 중...")
        self.processed_data = process_sensor_data(
            self.raw_data,
            missing_threshold=0.5,
            impute_method='knn',
            n_neighbors=5
        )
        
        # 결측치 통계 계산
        print("결측치 분석 중...")
        missing_stats = self.analyze_missing_values()
        
        # 그래프 생성
        print("그래프 생성 중...")
        self.generate_plots(output_dir)
        
        # 리포트 작성
        report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{output_dir}/analysis_report_{report_time}.txt"
        
        print("리포트 작성 중...")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 센서 데이터 분석 리포트 ===\n\n")
            f.write(f"분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"입력 파일: {self.csv_path}\n")
            f.write(f"파일 크기: {os.path.getsize(self.csv_path) / (1024 * 1024):.2f} MB\n\n")
            
            f.write("1. 데이터 기본 정보\n")
            f.write(f"- 전체 행 수: {self.raw_data.shape[0]}\n")
            f.write(f"- 전체 열 수: {self.raw_data.shape[1]}\n")
            f.write(f"- 데이터 기간: {self.raw_data.index.min()} ~ {self.raw_data.index.max()}\n\n")
            
            f.write("2. 결측치 분석\n")
            f.write(f"- 결측치가 있는 컬럼 수: {missing_stats['columns_with_missing']}\n")
            f.write("- 컬럼별 결측치 비율:\n")
            for col, pct in missing_stats['missing_percentages'].items():
                f.write(f"  * {col}: {pct:.2f}%\n")
            
            f.write("\n3. 데이터 처리 결과\n")
            f.write(f"- 제거된 컬럼 수: {len(self.raw_data.columns) - len(self.processed_data.columns)}\n")
            f.write(f"- 남은 컬럼 수: {len(self.processed_data.columns)}\n")
            f.write("- 처리된 데이터의 결측치 수: {}\n".format(self.processed_data.isnull().sum().sum()))
            
        self.report_path = report_path
        print(f"분석 리포트가 생성되었습니다: {report_path}")
        print(f"시각화 결과가 {output_dir} 디렉토리에 저장되었습니다.")

def analyze_sensor_data(csv_path: str, output_dir: str = "analysis_results", chunk_size: int = 10000):
    """
    센서 데이터 분석 실행 함수
    
    Args:
        csv_path: CSV 파일 경로
        output_dir: 결과물 저장 디렉토리
        chunk_size: 데이터를 읽을 때 사용할 청크 크기
    """
    analyzer = SensorDataAnalyzer(csv_path)
    if analyzer.load_data(chunk_size=chunk_size):
        analyzer.generate_report(output_dir)
        return analyzer.report_path
    return None

if __name__ == "__main__":
    # 사용 예시
    # csv_path = "sensor_data.csv"
    # chunk_size = 10000  # 메모리 상황에 따라 조절 가능
    # report_path = analyze_sensor_data(csv_path, chunk_size=chunk_size)
    pass 