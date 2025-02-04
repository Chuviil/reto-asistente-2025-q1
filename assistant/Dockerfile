FROM python:3.12-slim

EXPOSE 5000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=assistant.py

CMD ["flask", "run", "--host", "0.0.0.0"]
