import matplotlib
matplotlib.use('Agg')  # 서버 환경에서 그래프 그리기 위해 필요

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    error_msg = None
    
    if request.method == 'POST':
        try:
            # 1. 파일 및 K값 가져오기
            file = request.files['file']
            k_value = int(request.form.get('k_value', 3)) # 기본값 3
            
            if not file:
                raise ValueError("파일이 업로드되지 않았습니다.")

            # 2. 데이터 읽기 (CSV)
            df = pd.read_csv(file)
            
            # 3. 데이터 전처리
            # 시각화를 위해 숫자형 데이터만 선택하고, 결측치는 0으로 채움
            df_numeric = df.select_dtypes(include=[np.number]).fillna(0)
            
            # 2차원 평면 그래프를 그리기 위해 앞에서 2개의 컬럼만 사용
            if df_numeric.shape[1] < 2:
                raise ValueError("숫자형 데이터 컬럼이 최소 2개 이상 필요합니다.")
                
            X = df_numeric.iloc[:, :2].values # 첫 번째, 두 번째 컬럼만 사용
            feature_names = df_numeric.columns[:2] # 축 이름용

            # 4. K-means 클러스터링 수행
            kmeans = KMeans(n_clusters=k_value, random_state=42)
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_

            # 5. 그래프 그리기 (Matplotlib)
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 데이터 포인트 산점도
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
            
            # 중심점(Centroids) 표시
            ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')
            
            ax.set_title(f'K-means Clustering Result (K={k_value})')
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 6. 이미지를 메모리 버퍼에 저장 후 Base64로 인코딩
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close() # 메모리 해제

        except Exception as e:
            error_msg = str(e)

    return render_template('index.html', plot_url=plot_url, error_msg=error_msg)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)