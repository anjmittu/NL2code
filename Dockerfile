FROM python:2

RUN pip install vprof numpy theano astor h5py nltk

RUN python download_script.py

ENTRYPOINT ["/bin/bash"]
