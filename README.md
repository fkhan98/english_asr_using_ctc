# ASR_using_DL



## Methodology
ASR(Automatic Speech Recognition) is a problem that can be solved using deep learning. As there is no correspondence between the input and the output, we use CTC to model the solution to this problem. The code for the CTC based solution is in the dev-CTC branch. 

## Dataset
The dev-CTC branch has codes set up to trained on a subset of the Librispeech dataset(https://www.openslr.org/12). The dev-clean subset is used to train the model and tested on the test-clean subset.

## Model
The model that was trained on the subset of the Librispeech dataset was modeled from the deepspeech 2 model introduced in this paper(https://proceedings.mlr.press/v48/amodei16.pdf). The model architecture is defined in the asr_model.py script.

## Training 
The details about training the model are inside the train.py script. 


