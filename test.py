import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. 설정: 데이터 생성 개수
n_samples = 20000  # 2만 개 행
regions = ['Seoul', 'Busan', 'Incheon', 'Daegu', 'Daejeon', 'Gwangju']

print("데이터 생성 중... (상관관계 반영)")

# 2. 날짜 및 지역 데이터 생성
dates = [datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]
region_data = np.random.choice(regions, n_samples)

# 3. 미세먼지 농도 생성 (계절적 요인 반영 흉내 - 겨울/봄에 높게)
# 기본 농도에 랜덤 노이즈 추가
pm10 = np.random.normal(50, 20, n_samples)  # 평균 50, 표준편차 20
pm10 = np.maximum(pm10, 10) # 최소값 10

# 초미세먼지는 미세먼지의 약 50~70% 수준으로 생성
pm25 = pm10 * np.random.uniform(0.5, 0.7, n_samples)

# 4. 환자 수 데이터 생성 (미세먼지 농도와 양의 상관관계 부여)
# 기본 환자 수 + (미세먼지 영향) + 랜덤 변동
# 호흡기 질환: 미세먼지 영향 큼
respiratory_patients = np.random.randint(10, 50, n_samples) + (pm10 * 0.5) + (pm25 * 1.2)
respiratory_patients = respiratory_patients.astype(int)

# 심혈관 질환: 미세먼지 영향 중간
cardio_patients = np.random.randint(5, 30, n_samples) + (pm10 * 0.2) + (pm25 * 0.8)
cardio_patients = cardio_patients.astype(int)

# 알레르기성 비염: 미세먼지 영향 큼
rhinitis_patients = np.random.randint(20, 60, n_samples) + (pm10 * 0.6) + np.random.normal(0, 5, n_samples)
rhinitis_patients = np.maximum(rhinitis_patients, 0).astype(int)

# 5. 데이터프레임 만들기
df = pd.DataFrame({
    'Date': [d.strftime('%Y-%m-%d') for d in dates],
    'Region': region_data,
    'PM10_Concentration': np.round(pm10, 1),
    'PM2.5_Concentration': np.round(pm25, 1),
    'Respiratory_Patients': respiratory_patients, # 호흡기 질환
    'Cardio_Patients': cardio_patients,           # 심혈관 질환
    'Rhinitis_Patients': rhinitis_patients        # 비염
})

# 6. CSV 파일로 저장
file_name = 'fine_dust_health_data.csv'
df.to_csv(file_name, index=False)

print(f"완료! '{file_name}' 파일이 생성되었습니다.")
print(df.head())
print("-" * 30)
print(f"데이터 크기: {df.shape}")