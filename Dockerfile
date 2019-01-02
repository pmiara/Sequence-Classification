#FROM jupyter/minimal-notebook
FROM jupyter/tensorflow-notebook
COPY requirements.txt /tmp/
RUN mkdir /home/jovyan/.local
COPY setup.py /home/jovyan
RUN pip install --requirement /tmp/requirements.txt
RUN pip install -e .
