#FROM continuumio/anaconda
FROM sp0x/py_ml_bootstrap:latest
RUN mkdir -p /app

ARG source
WORKDIR /app

COPY app/requirements.txt /requirements.txt

RUN pip install -r /requirements.txt \
    && rm /requirements.txt
COPY /app /app

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s CMD curl -f http://localhost/status || exit 1

#Removed due to python 3 incompatability
#RUN conda install -c treeinterpreter  -y --quiet
EXPOSE 5556 5557 5560 80
CMD [ "-u", "__init__.py"]
ENTRYPOINT ["python"] 













































