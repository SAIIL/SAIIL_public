# Instructions:

Install miniconda from:
https://docs.conda.io/en/latest/miniconda.html

Install cuda as needed - we use cuda 10.0


Install conda environment 
```
conda env create -f ./env.saiil.yml -n saiil
```

Create protobuf wrappers
```
protoc --python_out=data_interface/ ./data_interface/sages.proto
```

TODO: update
