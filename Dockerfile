FROM python:3.12


WORKDIR /Case-Study-Teknasyon


COPY requirements.txt ./
COPY Model/xgb_model.pkl Model/
COPY Utils/ Utils/
COPY Datasets/ Datasets/
COPY Xgb-evaluate.py ./


RUN pip install --no-cache-dir -r requirements.txt


CMD ["python", "Xgb-evaluate.py"]