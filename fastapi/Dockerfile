FROM tiangolo/uvicorn-gunicorn:python3.7

RUN mkdir /fastapi

COPY requirements.txt /fastapi

COPY . /fastapi

WORKDIR /fastapi

RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html


EXPOSE 8008

CMD ["uvicorn", "server:app", "--host", "188.166.56.183", "--port", "8008"]