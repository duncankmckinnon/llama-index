FROM python:3.11 as build
ADD main.py .
RUN pip install python==3.11 arize==7.12.0 "arize-phoenix[evals]"
CMD ["python", "./main.py"]