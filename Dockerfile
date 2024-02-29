FROM python:3.11
COPY . /app
WORKDIR /app
# EXPOSE 2221
RUN apt-get update && apt update -y
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
CMD ["python3","app.py"]