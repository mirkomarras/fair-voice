# Fair Voice Toolbox
[![Build Status](https://travis-ci.org/pages-themes/cayman.svg?branch=master)](https://travis-ci.org/pages-themes/cayman)
[![GitHub version](https://badge.fury.io/gh/boennemann%2Fbadges.svg)](http://badge.fury.io/gh/boennemann%2Fbadges)
[![Dependency Status](https://david-dm.org/boennemann/badges.svg)](https://david-dm.org/boennemann/badges)
[![Open Source Love](https://badges.frapsoft.com/os/gpl/gpl.svg?v=102)](https://github.com/ellerbrock/open-source-badge/)

[Gianni Fenu](), [Mirko Marras](https://www.mirkomarras.com/), [Hicham Lafhouli](), [Giacomo Medda](), [Giacomo Meloni](www.linkedin.com/in/giacomo-meloni)

University of Cagliari

This repository contains a Python framework for bias mitigation analysis in Speaker Verification Models, in order to be able to verify and study what possible solutions may be to reduce the disparities in performance between sensitive categories such as gender and age.

This repository contains the source code of the following articles: 
- **"Exploring Algorithmic Fairness in Deep Speaker Verification"**, In Proc. of the International Conference on Computational Science and its Applications (ICSSA2020), Springer.
- **"Improving Fairness in Speaker Recognition"**, In Proc. of the Symposium on Pattern Recognition and Applications (SPRA 2020), Rome. 


![Screenshot](img/fair-voice_architecture.png)

## Table of Contents
- [Installation](#installation)
- [Fair-Voice Dataset](#fair-voice-dataset)
- [Fair-Voice Data Folder Description](#fair-voice-data-folder-organisation)
- [Pre-processing Module](#pre-processing-module)
- [Models](#models)
- [Train](#train)
- [Test](#test)
- [Experiment Utils](#experiment-utils)
    - [Equal Error Rate model's performance comparison](#equal-error-rate-model's-performance-comparison)
    - [Results Similarity Check](#similarity-check)
    - [Audio Features Similarity_Check](#audio-file-check)
- [Contribution](#contribution)
- [Citations](#citations)
- [License](#license)

       
## Installation

Install Python (>=3.7):
```
$ sudo apt-get update
$ sudo apt-get install python3.7
```
Install all the requirements needed:
```
$ pip install -r https://github.com/mirkomarras/fair-voice/blob/master/requirements.txt
```
Clone this repository:
```
$ git clone https://github.com/mirkomarras/fair-voice.git
```

## Fair-Voice Dataset
Our work is based on training and testing models using the Fair-Voice Dataset. This dataset was composed by downloading utterances from the Common Voice Mozilla public dataesets which can be found [here](https://commonvoice.mozilla.org/it/datasets).

The composite dataset was cleaned of any damaged audio tracks and organized by languages with enough samples to allow for proper processing. Subsequently, the audio files originally in MP3 format were converted to WAV format in order to make them suitable for the extraction of the features to be fed to the models.

To download the Fair-Voice dataset click [here](https://docs.google.com/forms/d/1Et3VxKpG2xKwOF46uT5sZvnTmOMiXYhVkZUJRwL7aFA/prefill) and follow the steps below:
1. Fill out the form you find via the link above;
2. Download, in the project folder, the zip of the dataset whose link will be provided via email;
3. Create the destination folder for the dataset and unzip the folder inside project:
    ```bash
    $ mkdir FairVoice/ && unzip file.zip -d FairVoice
    ```
    If the *unzip* command isn't already installed on your system, then run:
    ```bash
    $ sudo apt-get install unzip
    ```
4. Remove the .zip file: 
    ```bash
    $ sudo rm file.zip
    ```

## Fair-Voice Data Folder Description 

## Pre-processing Module

## Models
Based on the work done by our previous studies, we have made available a set of pre-trained models that can be downloaded [here](https://drive.google.com/drive/folders/1vRk5J4Nf-ohGRSnguPRkgNMh4i8n0UWB).
In this link you can find all the models trained from our two previous studies: 
- trained_model_iccsa   (*"Exploring Algorithmic Fairness in Deep Speaker Verification"*) [**Thin-ResNet34**];
- trained_model_spra    (*"Improving Fairness in Speaker Recognition"*) [**Thin-ResNet34**, **X-Vector**].

Each model needs to be copied into the appropriate folder ./exp/trained_model/. 

## Train
This toolbox allows you to train a speaker model from scratch. For instance, a Thin-ResNet model can be trained by running the following script and indicating the specific type of speaker model to train: 

```bash
$ python3 ./src/train.py --net resnet34vox
```

It is necessary to specify the CSV file that indicates at the script which audio file use for training. These .CSV files are stored in ./exp/train/ folder. For instance: 
```bash
$ python3 ./src/train.py --train_csv_path ./exp/test/English-train1.csv"
```

As we have seen, various parameters can be initialized for this script:
```
$ python train.py  --train_csv_path   "Path of the csv file that you want to use for train" 
                   --audio_dir        "Path of the directory that contains all the audio file"
                   --net              "Network architecture. Possible choices: [resnet34vox , xvector , resnet50vox , vggvo]"
                   --n_epochs         "Number of epochs"
                   --batch            "Batch size"
                   --learning_rate    "Learning rate "
                   --decay_factor     "Decay factor"
                   --decay_step       "Decay step"
                   --loss             "Loss"
                   --aggregation      "Aggregation strategy"
                   --vlad_cluster     "Number of Vlad Cluster"
                   --ghost_clusters   "Number of Ghost Cluster"
```

It is possibile to customize the default choice for this parameters by modify the train script changing the *default* parameter, in order to optimize the use of the script:
```python
parser.add_argument('--net', dest='net', default='resnet34vox', type=str, action='store', help='Network model architecture')
```

The training script will save the model each epoch in ./exp/trained_model/ in a folder with the following formatting name:

> NETWORK-NAME_DDMMYYYY_hh:mm_INFO-DATASET

Inside each folder is saved a "model.h5" file per epoch. 

This toolbox manage several speaker verification models: 

| Model                     | Input         | Shape         |
| -------------             | ------------- | ------------- |
| ResNet34vox (Thin-ResNet) | Spectrogram   | (256,None,1)  |
| X-Vector                  | Filter-banks  | (None,24)     |
| ResNet50vox               | Spectrogram   | (256,None,1)  |
| VggVox                    | Spectrogram   | (256,None,1)  |

## Test
This toolbox provides a script to test a pre-trained model in order to elaborate subsequently different metrics on the basis of the results. For instance, a Thin-ResNet model can be tested by running the following script and indicating the specific type of speaker model to train and the path of the pre-trained model:
```bash
$ python3 ./src/test.py --net resnet34vox --model_path "./exp/trained_model/resnet34vox_18062020_2334_English-Spanish-train3/weights-15-0.993.h5"
```

It is necessary to specify the CSV file that indicates at the script which audio file use for testing. These .CSV files are stored in ./exp/test/ folder. For instance:  
```bash
$ python3 ./src/test.py --test_file ./exp/train/English-test1.csv"
```

As we have seen, various parameters can be initialized for this script:
  
```
$ python test.py  --net              "Network architecture. Possible choices: [resnet34vox , xvector , resnet50vox , vggvo]"
                --test_file        "Path of the csv file that we want to use for our tests"
                --audio_dir        "Path of the directory that contains all the audio file"
                --model_path       "Path of the h5 file saved frome the training session and used to load the model"
                --aggregation      "Aggregation strategy"
```

It is possibile to customize the default choice for this parameters by modify the test script changing the *default* parameter, in order to optimize the use of the script:

```python
parser.add_argument('--net', dest='net', default='resnet34vox', type=str, action='store', help='Network model architecture')
```

Every test will save a .CSV file in the folder ./exp/results/ with the following format:
> NETWORK-ARCHITECTURE_INFO-DATASET-TRAIN_ACCURACY_DDMMYYYY_hh:mm_INFO-DATASET-TEST

Every result file contains the following information: 
```
> audio1            path of the first audio utterance
> audio2            path of the second audio utterance
> age1              age info of the first speaker ('young' <= 40yo, 'old' > 40yo)
> age2              age info of the second speaker ('young' <= 40yo, 'old' > 40yo)
> gender1           gender info of the first speaker ('male', 'female')
> gender2           gender info of the second speaker ('male', 'female')
> label             expeted result
> similarity        predicted result
```

## Experiment utils
Our framework allows, on the basis of the results processed, to calculate a series of metrics useful for comparing the performance of the trained models, especially from the point of view of the disparity between the sensitive categories considered (gender, age).

All the scripts related to these calculations are inside the ./exp_utils/ folder.

#### Metrics model's performance comparison
To compare the performance of the trained models using metric like EER (*Equal Error Rate*), FAR (*False Acceptance Rate*), FRR(*False Rejection Rate*), both in general and individually for each group of users (male, female, young, old), it is possible to use the script *exp_utils.py*:

```bash
$ python exp_utils.py
```
The script takes two different parameters:
```
$ python exp_utils.py   --result_path        "Results folder path"
                        --dest_folder        "Destination path to save the computed metrics"
```
By default the result path is set in ./exp/results/ folder. 

The script processes all .CSV files inside the results folder and return as output three .CSV files (EER.csv, FAR.csv, FRR.csv) with all the calculated metrics for each result file.

By default the metric file is stored inside ./exp/metrics/. Each time this script is run, the contents of this folder are deleted and all results currently contained in the *result folder* are retried. 

In every metric file are specified common information concerning the architecture, the train and test datasets used for experiments and finally the accuracy of the considered model.
```
> Architecture            path of the first audio utterance
> Train File            path of the second audio utterance
> Test File              age info of the first speaker ('young' <= 40yo, 'old' > 40yo)
> Accuracy              age info of the second speaker ('young' <= 40yo, 'old' > 40yo)
```

Based on the metric, information about the general performance and the performance of the individual sensitive categories will then be reported for each result file.

#### Similarity Check
Thanks to the script *statistic_distribution.py* it is possible to check if the reported results present a similar distribution between two categories results in order to understand if the results are co-related or not.

A supporting .JSON file named '*path_best_results.json*' is used to process a specific set of results: inside it the paths of the best results to be compared are inserted, divided by defined tests.

```json
{
  "English-test1": [
    "/home/meddameloni/dl-fair-voice/exp/results/xvector_English-train1_991_11062020_English-test1.csv",
    "/home/meddameloni/dl-fair-voice/exp/results/xvector_English-train2_962_11062020_English-test1.csv"
  ],
  "English-test2": [
    "/home/meddameloni/dl-fair-voice/exp/results/xvector_English-train1_969_11062020_English-test2.csv",
    "/home/meddameloni/dl-fair-voice/exp/results/xvector_English-train2_987_11062020_English-test2.csv"
  ]
}
```

Once this file has been defined, it will be possible to execute the script by specifying one of the following parameters:

```
$ python statistic_distribution.py      --eer   statistical correlations are processed regarding EER results
                                        --far   statistical correlations are processed regarding FAR results
                                        --frr   statistical correlations are processed regarding FRR results
```

Depending on your preference, statistical correlations will be calculated based on a specific metric. These processing are done only on the files specified in the JSON.

If the processing is done for *EER* a .CSV file is generated in the output folder ./exp/statistics/ where for each result file it is specified whether the results of the related sensitive categories are statistically similar with a 'Y' or an 'N'.

If the processing is done for *FAR* or *FRR* then a .CSV file is generated in the output folder ./exp/statistics/ where for each result file it is specified what we said above and the report of total cases by sensitive category grouped by user is added.


## Contribution
This code is provided for educational purposes and aims to facilitate reproduction of our results, and further research
in this direction. We have done our best to document, refactor, and test the code before publication.

If you find any bugs or would like to contribute new models, training protocols, etc, please let us know.

Please feel free to file issues and pull requests on the repo and we will address them as we can.

## Citations

```
Fenu, G., Lafhouli, H. and Marras, M. (2020).
Exploring Algorithmic Fairness in Deep Speaker Verification.
In: International Conference on Computational Science and its Applications (ICSSA2020)
```

```
Fenu, G., Marras, M., Medda, G. and Meloni, G. (2020).
Improving Fairness in Speaker Recognition.
In: Symposium on Pattern Recognition and Applications (SPRA 2020) 
```

## License
This code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This software is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the GNU General Public License for details.

You should have received a copy of the GNU General Public License along with this source code. If not, go the following link: http://www.gnu.org/licenses/.
