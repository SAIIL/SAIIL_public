# Instructions:

## Install miniconda from:
https://docs.conda.io/en/latest/miniconda.html

Install cuda as needed - we use cuda 10.0

## Install conda environment 
```
conda env create -f ./env.saiil.yml -n saiil
```

## Create protobuf wrappers
```
protoc --python_out=data_interface/ ./data_interface/sages.proto
```

## Convert cholec80 to protobuf
```
python ~/SAIIL_public/src/data_interface/cholec_convert.py  ~/SAIIL_public/data/annotations/deidentified/cholec80_protobuf/ --phase-folder  ~/SAIIL_public/data/cholec80/phase_annotations -v --tool-folder ~/SAIIL_public/data/cholec80/tool_annotations
```

## Train an example phase classification network
```
~/SAIIL_public/src/scripts/run_script_temporal_model.sh
```
