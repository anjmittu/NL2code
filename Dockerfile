FROM python:2

RUN pip install vprof numpy theano astor h5py nltk

ENTRYPOINT ["/bin/bash"]
