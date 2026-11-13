# ML Churn Service
Сервис на FastAPI для предсказания оттока клиентов (churn) по табличным данным.

## Запуск
```bash
conda activate churn
pip install -r requirements.txt
uvicorn app.main:app --reload

Swagger: http://127.0.0.1:8000/docs

Docker
docker build -t churn-service:latest .
docker run --rm -p 8000:8000 churn-service:latest

Как проверить
POST /model/train — обучить модель на data/churn_dataset.csv
GET /health — проверить, что модель и датасет загружены (status: ok)
POST /predict — получить предсказание churn и вероятности

Тесты
pytest