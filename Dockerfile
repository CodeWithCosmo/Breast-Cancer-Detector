# FROM python:3.8
# COPY . /app
# WORKDIR /app
# RUN pip install -r requirements.txt 
# EXPOSE $8000
# CMD gunicorn -w 4 -b 0.0.0.0:$8000 app:app

FROM python:3.8

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["waitress-serve", "--listen=*:8080", "app:app"]
