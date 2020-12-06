# SAIIL_public

This folder containts SAIIL_public demo code for SAGES protobuf standard. The demo includes:
* Converter from [cholec80 data](http://camma.u-strasbg.fr/datasets) to protobuf for phases and tools.
* The training of a phase segmentation network based on this data. The network has an Resnet18 visual model and LSTM temporal model.

In addition under the [Poemnet](poemnet/) directory -- the directory contains the code and examples in order to perform the
analyses in our paper "Automated operative phase identification in
peroral endoscopic myotomy", DOI 10.1007/s00464-020-07833-9:
* Segment annotation (anvil) files.
* Supporting statistics code.
 

# Instructions for setting up SAGES protobuf SDK:

## Install miniconda from:
https://docs.conda.io/en/latest/miniconda.html

Install cuda as needed - we use cuda 10.0, on Ubuntu 18.04.

## Install conda environment 
```
conda env create -f ~/SAIIL_public/src/env.saiil.yml -n saiil
```

## Create protobuf wrappers
```
cd ~/SAIIL_public/src
protoc --python_out=./ ./data_interface/sages.proto
```

## Convert cholec80 to protobuf
```
python ~/SAIIL_public/src/data_interface/cholec_convert.py  ~/SAIIL_public/data/annotations/cholec80_protobuf/ --phase-folder  ~/SAIIL_public/data/cholec80/phase_annotations -v --tool-folder ~/SAIIL_public/data/cholec80/tool_annotations
```

## Train an example phase classification network
```
~/SAIIL_public/src/scripts/run_script_temporal_model.sh
```
