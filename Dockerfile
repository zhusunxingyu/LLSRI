FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements_public.txt ./requirements_public.txt
RUN pip install --no-cache-dir -r requirements_public.txt

COPY . .

EXPOSE 8088

CMD ["python", "public_web_server.py"]
