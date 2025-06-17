# Mult-NNs
A pair of neural networks predicting the result of multiplication of two numbers.
## Repo structure:
All code is located in the “/src” folder.

The folder “/src/hundreds” contains a model trained mostly on values x ∈ [0, 100] and it's training data.  It is good at predicting numbers up to and including 100.

The folder “/src/thousands” contains a model trained mainly on values x ∈ [100, 1000].  It is good at predicting numbers from 100 up to and including 1000.

For values beyond the training limits, the scatter of output values is **very large**.

##Running the models

To run a model on your machine, follow the Pytorch documentation for [loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

Or you can just use “loader.py”. Before use replace the MAX_VAL_EXP value with a degree of ten of the maximum training set value.(2 for hundreds and 3 for thousands).
