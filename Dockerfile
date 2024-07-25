FROM python:3.6.6

RUN groupadd -r multiclasser && useradd -r -g multiclasser multiclasser
WORKDIR /home/multiclasser/

ADD req.txt req.txt
RUN pip install -r req.txt

ADD multiclass .
ADD entrypoint.sh entrypoint.sh

RUN chown -R multiclasser:multiclasser /home/multiclasser
RUN chmod a+x entrypoint.sh

USER multiclasser

ENTRYPOINT ["./entrypoint.sh"]
