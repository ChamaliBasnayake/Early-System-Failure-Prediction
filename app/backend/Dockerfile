FROM python:3.11-slim
WORKDIR /srv
ENV PIP_NO_CACHE_DIR=1
RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install pandas==2.2.2 pyarrow==17.0.0 numpy==1.26.4 drain3==0.9.11 fastapi==0.112.2 "uvicorn[standard]==0.30.6" sqlalchemy==2.0.32 psycopg2-binary==2.9.9 pydantic==2.8.2 pyyaml==6.0.2 scikit-learn==1.5.1
COPY src ./src
COPY configs ./configs
COPY app/__init__.py ./app/__init__.py
COPY app/backend ./app/backend
# model artefacts are mounted at runtime (see docker-compose.yml): data/interim (drain state, texts) and results (checkpoints, scores)
EXPOSE 8000
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
