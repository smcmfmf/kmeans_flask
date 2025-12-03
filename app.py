from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import io
import base64
import werkzeug

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

plt.rcParams['axes.unicode_minus'] = False

@app.errorhandler(werkzeug.exceptions.RequestEntityTooLarge)
def handle_file_too_large(e):
    return render_template('index.html', error_msg="파일이 너무 큽니다. 최대 50MB까지 업로드 가능합니다."), 413

@app.route('/preview', methods=['POST'])
def preview():
    try:
        if 'file' not in request.files:
            return {"error": "No file provided"}, 400
        f = request.files['file']
        try:
            f.stream.seek(0)
        except Exception:
            pass

        raw = f.read()
        try:
            text = raw.decode('utf-8')
        except Exception:
            try:
                text = raw.decode('cp949')
            except Exception:
                text = raw.decode('utf-8', errors='replace')

        df = pd.read_csv(io.StringIO(text), nrows=5, on_bad_lines='skip', low_memory=False)
        html = df.to_html(classes="table table-sm", index=False)
        return {"html": html}
    except Exception as ex:
        app.logger.exception("Preview failed")
        return {"error": f"{str(ex)}"}, 500

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    error_msg = None
    method_text = ""
    k_value = 3

    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return render_template('index.html', error_msg="No file uploaded.")
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', error_msg="Please select a file.")

            try:
                file.stream.seek(0)
            except Exception:
                pass

            raw = file.read()
            try:
                file.stream.seek(0)
            except Exception:
                pass

            try:
                text = raw.decode('utf-8')
            except Exception:
                try:
                    text = raw.decode('cp949')
                except Exception:
                    text = raw.decode('utf-8', errors='replace')

            try:
                df = pd.read_csv(io.StringIO(text), low_memory=False, on_bad_lines='skip')
            except Exception as e_read:
                app.logger.exception("pd.read_csv failed")
                return render_template('index.html', error_msg="Cannot read CSV file. (파싱 실패)")

            df_numeric = df.select_dtypes(include=[np.number]).fillna(0)

            if df_numeric.shape[1] < 2:
                return render_template('index.html', error_msg="Need at least 2 numeric columns.")

            if df_numeric.shape[0] > 100000:
                df_numeric = df_numeric.sample(100000, random_state=42)

            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df_numeric)

            graph_title = ""

            method = request.form.get('method', 'std')
            k_value = int(request.form.get('k_value', 3))

            if method == 'pca':
                pca = PCA(n_components=2)
                pca_data = pca.fit_transform(scaled_features)

                cluster_data = pca_data
                plot_data = pca_data

                method_text = "PCA 차원축소 후 클러스터링"
                graph_title = "K-means Clustering (PCA Reduction)"
                x_label, y_label = "PCA Component 1", "PCA Component 2"

            else:
                cluster_data = scaled_features

                if scaled_features.shape[1] > 2:
                    pca = PCA(n_components=2)
                    plot_data = pca.fit_transform(scaled_features)
                    x_label, y_label = "PCA Component 1 (Vis)", "PCA Component 2 (Vis)"
                else:
                    plot_data = scaled_features
                    x_label, y_label = df_numeric.columns[0], df_numeric.columns[1]

                method_text = "표준화된 전체 데이터 클러스터링"
                graph_title = "K-means Clustering (Standardized Data)"

            kmeans = KMeans(n_clusters=k_value, random_state=42)
            clusters = kmeans.fit_predict(cluster_data)

            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(plot_data[:, 0], plot_data[:, 1], c=clusters,
                                  cmap='viridis', s=50, alpha=0.8)

            plt.title(f'{graph_title} (K={k_value})', fontsize=14)
            plt.xlabel(x_label, fontsize=11)
            plt.ylabel(y_label, fontsize=11)
            plt.colorbar(scatter, label='Cluster ID')
            plt.grid(True, linestyle='--', alpha=0.5)

            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

        except Exception as e:
            app.logger.exception("Index processing failed")
            error_msg = f"Error: {str(e)}"

    return render_template('index.html', plot_url=plot_url, error_msg=error_msg,
                           method_text=method_text, k_value=k_value)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
