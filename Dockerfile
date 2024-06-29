

####### 👇 SIMPLE SOLUTION (x86 and M1) 👇 ########
FROM python:3.8.12-buster

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY hr_data_analytics_package_folder hr_data_analytics_package_folder
COPY setup.py setup.py
RUN pip install .

COPY raw_data raw_data

COPY Makefile Makefile

CMD uvicorn hr_data_analytics_package_folder.api:app --host 0.0.0.0 --port $PORT
