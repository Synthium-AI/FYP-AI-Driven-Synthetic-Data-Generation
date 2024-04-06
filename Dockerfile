FROM python:3.11 AS build

WORKDIR /app


# RUN pip install --trusted-host pypi.python.org -r requirements.txt
ADD ./requirements.txt /app/requirements.txt
RUN pip3 install --default-timeout=500 torch
RUN pip3 install --default-timeout=500 -r requirements.txt

COPY . /app

# Run Service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
