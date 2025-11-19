FROM python:3.11-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# Instala todo de una sola vez
RUN apt-get update && apt-get install -y procps && \
    pip install -U pip && \
    pip install -r ./requirements.txt && \
    rm /etc/localtime && \
    ln -s /usr/share/zoneinfo/America/Mexico_City /etc/localtime

COPY ./api.py /code/

EXPOSE 8000

CMD ["uvicorn","api:app", "--host", "0.0.0.0", "--port", "8000"]