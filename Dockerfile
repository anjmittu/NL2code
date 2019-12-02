FROM python:2

RUN pip install vprof numpy theano astor h5py nltk

RUN python -m nltk.downloader punkt

ENTRYPOINT ["/bin/bash"]
