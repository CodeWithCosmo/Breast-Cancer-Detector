FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt 
EXPOSE $8000
# CMD gunicorn -w 4 -b 0.0.0.0:$8000 app:app
CMD ["waitress-serve", "--listen=*:8080", "app:app"]
