from flask import Flask, request, jsonify, render_template
import torch
from src.model.NRMS import NRMS  # Import lớp NRMS từ __init__.py đã cung cấp
import pandas as pd

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Thiết lập thiết bị để chạy mô hình (GPU nếu có, nếu không thì sử dụng CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cấu hình cho mô hình
class Config:
    num_words = 10000
    word_embedding_dim = 300
    num_attention_heads = 15
    query_vector_dim = 200
    num_categories = 50
    category_embedding_dim = 100
    dropout_probability = 0.2

config = Config()

# Khởi tạo mô hình NRMS với word embedding được khởi tạo ngẫu nhiên
model = NRMS(config).to(device)
model.eval()

# Tải dữ liệu từ file news.tsv
news_file_path = r"C:\Users\trong\Documents\Final-news_recommendation\news-recommendation\src\data\test\news.tsv"
news_data = pd.read_csv(news_file_path, sep='\t', header=None)

# Cập nhật tên các cột phù hợp với số lượng cột thực tế trong file news.tsv
news_data.columns = ['ID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Entities', 'Relations']

# Tạo route cho trang chủ
@app.route('/')
def home():
    # Chuyển dữ liệu thành danh sách dictionary để truyền vào giao diện
    news_list = news_data[['Title', 'Category', 'URL']].to_dict(orient='records')
    return render_template('index.html', news_list=news_list)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    candidate_news = data.get('candidate_news')
    clicked_news = data.get('clicked_news')

    # Chuyển đổi dữ liệu đầu vào thành tensor
    candidate_news_tensor = [convert_news_to_tensor(news, config) for news in candidate_news]
    clicked_news_tensor = [convert_news_to_tensor(news, config) for news in clicked_news]

    # Tạo dự đoán
    with torch.no_grad():
        click_probability = model(candidate_news_tensor, clicked_news_tensor)

    return jsonify({'click_probability': click_probability.tolist()})

def convert_news_to_tensor(news, config):
    """
    Chuyển đổi dữ liệu news thành tensor theo cấu hình của mô hình.
    """
    return {
        "title": torch.tensor(news["title"]).to(device),
        "abstract": torch.tensor(news["abstract"]).to(device),
        "category": torch.tensor(news["category"]).to(device),
        "subcategory": torch.tensor(news["subcategory"]).to(device),
    }

if __name__ == '__main__':
    app.run(debug=True)
