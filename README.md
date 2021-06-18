# Useful scripts to use DeepFrag


This set of scripts can be used to make a prediction using the pre-trained CNN [DeepFrag](https://pubs.rsc.org/en/content/articlelanding/2021/sc/d1sc00163a#!divAbstract) with custom fragment libraries. 

## Install DeepFrag
DeepFrag package is needed for running a prediction. 

```
# Clone DeepFrag python package.
!git clone https://git.durrantlab.pitt.edu/jdurrant/deepfrag.git
```
## Create a custom library

DeepFrag needs a fragments library in fingerprints to be able to select the top fragments according to the prediction. One library is already provided by the DeepFrag autohrs: 
- 5564 fragments.

But, additionally, more fragments can be added from PDB files to this library or a different library can be build. 

To add fragments to their library, 

```
python generate_library.py path\to\fragments\folder
```

And, to create a different library, 
```
python generate_library.py path\to\fragments\folder --only
```
## Prediction

To run a prediction, 
```
python prediction.py \path\to\fingerprints\library path\to\receptor path\to\parent -g connectivity
```
