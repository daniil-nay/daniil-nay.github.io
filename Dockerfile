FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends nginx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Копируем наш конфиг nginx
COPY nginx.conf /etc/nginx/nginx.conf

# Открываем порт
EXPOSE 80

# Запускаем nginx на переднем плане
CMD ["nginx", "-g", "daemon off;"]

#CMD ["python", "main.py"]
