# Atone

This is a python module for the Kaggle DECMEG competition and sorrounding
problem statement of capturing visual stimuli from brain EEG/MEG, also known
as "decoding". The module further contains functionality to classify potential
outliers in the data, such as subjects suffering from visual agnosia or
prosopagnosia.

## Requirements

This module requires python3 in order to compile. Other dependencies of the
module can be found within the `requirements.txt` file. These can in turn
easily be installed by running `python3 install -requirements requirements.txt`.

## Usage

Run `python3 setup.py install` to install the module.

The atone package should now be accessible systemwide for the version of
python you installed it for.

To run the example pipelines provided by the module you can do the following:
`python3 -m examples/pooling.py`.

Do note however, that the folders `train` and `test` must be present in the
data folder or in an otherwise specified folder.

## Provided examples

The signal classification systems included are as follows.

### Pooling

Simple pooling over all channels for the direct time after stimulus.

### N170

Filterbanking for the gamma band.
