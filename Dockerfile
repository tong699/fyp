FROM python:3.9-slim
WORKDIR /app

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY rl_service.py /app
COPY actions.csv /app

EXPOSE 8000

CMD ["uvicorn", "rl_service:app", "--host", "0.0.0.0", "--port", "8000"]
