FROM python:3.7

RUN pip install vprof numpy theano astor h5py nltk

ENTRYPOINT ["/bin/bash"]
