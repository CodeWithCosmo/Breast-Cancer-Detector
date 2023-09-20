FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt 
EXPOSE $PORT
CMD gunicorn -w 2 -b 0.0.0.0:$PORT app:app