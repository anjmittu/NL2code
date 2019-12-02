# NL2code

A syntactic neural model for parsing natural language to executable code [paper](https://arxiv.org/abs/1704.01696).

## Dataset and Trained Models

Get serialized datasets and trained models from [here](https://drive.google.com/drive/folders/0B14lJ2VVvtmJWEQ5RlFjQUY2Vzg). Put `models/` and `data/` folders under the root directory of the project.

## Setting up Docker
Build the docker image
```
$ docker build -t intent2code .
$ docker run -it --rm -v `pwd`:/usr/src/myapp -w /usr/src/myapp intent2code
```

If you are using Docker to run, and the process is killed while running, this may be because you need
to increase the CPU and Memory docker has access to.  The defaults for docker are normally not enough.

## Usage

To train new model

```bash
. train.sh [hs|django]
```

To use trained model for decoding test sets

```bash
. run_trained_model.sh [hs|django]
```

## Dependencies

* Theano
* vprof
* NLTK 3.2.1
* astor 0.6

## Converting CoNaLa data to be used
The CoNaLa dataset can be downloaded [from this website](https://conala-corpus.github.io/0).  This data should be added under the folder `./lang/conala/data`.  

If you are not using the docker image, you will also need to run the following command:
```
$ python -m nltk.downloader punkt
```

Once the data is added, the bin file can be created by running:
```
$ python lang/conala/convert_dataset.py
```

The file will be added to the data directory.

## Reference

```
@inproceedings{yin17acl,
    title = {A Syntactic Neural Model for General-Purpose Code Generation},
    author = {Pengcheng Yin and Graham Neubig},
    booktitle = {The 55th Annual Meeting of the Association for Computational Linguistics (ACL)},
    address = {Vancouver, Canada},
    month = {July},
    url = {https://arxiv.org/abs/1704.01696},
    year = {2017}
}
```
