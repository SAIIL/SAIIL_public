# SAIIL_public repository

Code under [src/](./src/) include: 

* SAGES protobuf demo/SDK code and phase segmentation model training, following the SAGES video annotation workshop, Houston, 2020. 
* Under the [Poemnet](./src/poemnet) folder - the code and examples in order to perform the
analyses in our paper "Automated operative phase identification in
peroral endoscopic myotomy", DOI 10.1007/s00464-020-07833-9.


### SAGES Protobuf Demo/SDK

The [src] containts SAIIL_public demo/SDK code for SAGES protobuf standard. The demo includes:
* Converter from [cholec80 data](http://camma.u-strasbg.fr/datasets) to protobuf for phases and tools.
* The training of a phase segmentation network based on this data. The network has an Resnet18 visual model and LSTM temporal model.

### Poemnet paper additional code
In addition under the [Poemnet](poemnet/) directory -- the directory contains the code and examples in order to perform the
analyses in our paper "Automated operative phase identification in
peroral endoscopic myotomy", DOI 10.1007/s00464-020-07833-9:
* Segment annotation (anvil) files.
* Supporting statistics code.
 
## Instructions for setting up SAGES protobuf SDK:

### Clone the repository

```
cd ~
git clone git@github-personal:SAIIL/SAIIL_public.git
```
### Install miniconda environment for the SDK
* Install cuda as needed - we assume cuda 10.0, Ubuntu 18.04.
* Set environment variable for repository location: We assume
```
export SAIIL_PUBLIC=~/SAIIL_public
```
* Install miniconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).
* Install conda environment:
```
conda env create -f ${SAIIL_PUBLIC}/src/env.saiil.yml -n saiil
```

### Create protobuf wrappers
```
cd ${SAIIL_PUBLIC}/src
protoc --python_out=./ ./data_interface/sages.proto
```

### Convert cholec80 to protobuf
* Download Cholec80 from the [Cholec80 website](http://camma.u-strasbg.fr/datasets) at Uni. of Strasbourg.
* Convert the Cholec80 data using the following command:
```
python ${SAIIL_PUBLIC}/src/data_interface/cholec_convert.py  ${SAIIL_PUBLIC}/data/annotations/cholec80_protobuf/ --phase-folder  ${SAIIL_PUBLIC}/data/cholec80/phase_annotations -v --tool-folder ${SAIIL_PUBLIC}/data/cholec80/tool_annotations
```

### Train an example phase classification network
Run:
```
${SAIIL_PUBLIC}/src/scripts/run_script_temporal_model.sh
```
(Verify that the folder names match your repository clone, and that you have compiled the protobuf, and downloaded/converted cholec80 data to protobuf)

The code includes a phase classification network with:
* Resnet-finetuned visual model.
* LSTM temporal model.
* Pytorch dataset/loader based on the protobufs defined in the SAGES 20' video/data annotation workshop in Houston.

The main training script is under [src/scripts/train_temporal_model.py](scripts/train_temporal_model.py).
