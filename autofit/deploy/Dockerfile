FROM python:3.9-slim

COPY ./MIWs_AutoFit.html /deploy_net/
COPY ./requirements.txt /deploy_net/
COPY ./autofit-slim /deploy_net/autofit-slim

COPY ./*.py /deploy_net/


WORKDIR /deploy_net/

RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "MIWs_AutoFit_flask_API.py"]
