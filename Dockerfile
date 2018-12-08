#FROM jupyter/minimal-notebook
FROM jupyter/tensorflow-notebook
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
RUN mkdir /home/jovyan/.local