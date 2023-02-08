# Causal Reasoning for Algorithmic Fairness in Voice Controlled Cyber-Physical Systems

[Gianni Fenu](https://people.unica.it/giannifenu/), [Hicham Lafhouli](), [Giacomo Medda](https://www.linkedin.com/in/giacomo-medda-4b7047200), [Giacomo Meloni](https://www.linkedin.com/in/giacomo-meloni), [Mirko Marras](https://www.mirkomarras.com/)

University of Cagliari

This branch expands the main repository with the materials used for the following article:
- **"Causal Reasoning for Algorithmic Fairness in Voice Controlled Cyber-Physical Systems"** submitted at the *Special Issue on Pattern Recognition for Cyber-Physical-Social Services* of the *Pattern Recognition Letters* journal.

This article and its source code propose an explanatory framework that aims to provide a better understanding of how speaker recognition models perform as the underlying voice characteristics on which they are tested change. With our framework, we evaluate two state-of-the-art speaker recognition models, comparing their fairness in terms of security, through a systematic analysis of the impact of more than twenty voice characteristics.

![schema](https://user-images.githubusercontent.com/26059819/185790147-ca8ff0d3-2765-4ab2-9d66-3501571d9b65.png)

## Table of Contents
- [Pre-requirements](#pre-requirements)
- [Data Preprocessing](#data-preprocessing)
- [Audio Features Extraction](#audio-features-extraction)
- [Model Evaluation](#model-evaluation)
- [Causal Classifier Training](#causal-classifier-training)
- [Causal Classifier Analysis](#causal-classifier-analysis)
- [Paper Results](#paper-results)
- [Citations](#citations)
- [License](#license)

       
## Pre-requirements

Before using the source code to reproduce the article, you should install the [FairVoice toolbox](https://github.com/mirkomarras/fair-voice) according to the README of the main repository. Once done, you should download the FairVoice dataset following the same document and train a deep speaker recognition architecture (ResNet-34, X-Vector etc.) with a training set. In our experiments we used models trained on the multi-language Train-1, i.e. English-Spanish Train-1 generated with the protocol presented in **"Improving Fairness in Speaker Recognition"**, In Proc. of the Symposium on Pattern Recognition and Applications (SPRA 2020), Rome.

## Data Preprocessing

First, we need to create a testing set on the basis of the training set selected and other parameters. The [pre-processing script](https://github.com/mirkomarras/fair-voice/blob/feature/audio_feat_ext/src/data/preprocessing.py) accepts several parameters, e.g. the number of negative and positive pairs. The following command creates the testing set used in our experiments:
```
python3 preprocessing.py --metadata_path /BASE_PATH/FairVoice/metadata.csv --languages English Spanish --n_users 75 --min_sample 6 --needed_users_path /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/preprocessing_data/ENG_SPA_1_TRAIN_users_dict.pkl --output_metadata_path /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/preprocessing_data/metadata_ENG_SPA_75users_6minsample.csv --neg_pairs 50 --pos_pairs 5 --fairvoice_path /BASE_PATH/FairVoice --samples_per_user 6 --output_test_path /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/preprocessing_data/test_ENG_SPA_75users_6samples_50neg_5pos.csv
```

## Audio Features Extraction

The testing set just created comprises audio samples, which need to be manipulated to extract the features to feed in input
to the explanatory framework. The [audio features extractor script](https://github.com/mirkomarras/fair-voice/blob/feature/audio_feat_ext/src/helpers/audio_features.py)
accepts several parameters, that are described inside the script. The features to extract are listed in the same script
and can be extracted as follows:
```
python3 audio_features.py --fairvoice_path /BASE_PATH/FairVoice --test_path /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/preprocessing_data/test_ENG_SPA_75users_6samples_50neg_5pos.csv --metadata_path /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/preprocessing_data/metadata_ENG_SPA_75users_6minsample.csv --save_path /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/audio_analysis/test_new_audio_features.pkl 
```

## Model Evaluation

The models should now be evaluated to extract the metric values for the dependent variable, e.g. false acceptance rate (FAR),
that generate pickle files with the far data mapped as 0 and 1 depending on the selected threshold for FAR (or false rejection rate, FRR). For
each model the [script](https://github.com/mirkomarras/fair-voice/blob/feature/audio_feat_ext/src/evaluation/evaluate.py) is run as follows:
```
Resnet34
python3 evaluate.py --results_file /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/results/resnet34vox_English-Spanish-train1@15_920_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10.csv --plots_kde_filepath "/BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/evaluation/plots/{}_kde__resnet34vox_English-Spanish-train1@15_920_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10.png" --far_label_0_le 0 --frr_label_0_le 0 --plots_hist_filepath "/BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/evaluation/plots/{}_0_0_hist__resnet34vox_English-Spanish-train1@15_920_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10.png"

X-Vector
python3 evaluate.py --results_file /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/results/xvector_English-Spanish-train1@13_992_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10.csv --plots_kde_filepath "/BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/evaluation/plots/{}_kde__xvector_English-Spanish-train1@13_992_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10.png" --far_label_0_le 0 --frr_label_0_le 0 --plots_hist_filepath "/BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/evaluation/plots/{}_0_0_hist__xvector_English-Spanish-train1@13_992_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10.png"
```

## Causal Classifier Training

Now we have all the data necessary to train the causal classifier to find the importance of each audio feature to predict
the dependent variable, FAR or FRR. The [training script](https://github.com/mirkomarras/fair-voice/blob/feature/audio_feat_ext/src/helpers/causal_classifier.py)
can be executed for both each speaker recognition architecture with Random Forest (RF) as follows:
```
ResNet-34
python3 causal_classifier.py --af_path /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/audio_analysis/test_new_audio_features.pkl --al_path /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/evaluation/far_data__resnet34vox_English-Spanish-train1@15_920_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10.pkl --metr_feats_folder /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/classification/far_data__resnet34vox_English-Spanish-train1@15_920_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10 --dataset_save_path /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/classification --model_save_path /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/classification/far_data__resnet34vox_English-Spanish-train1@15_920_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10/causal_classifier_RF.model --cc RF
    
X-Vector
python3 causal_classifier.py --af_path /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/audio_analysis/test_new_audio_features.pkl --al_path /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/evaluation/far_data__xvector_English-Spanish-train1@13_992_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10.pkl --metr_feats_folder /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/classification/far_data__xvector_English-Spanish-train1@13_992_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10 --dataset_save_path /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/classification --model_save_path /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/classification/far_data__xvector_English-Spanish-train1@13_992_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10/causal_classifier_RF.model --cc RF
```

## Causal Classifier Analysis

The [causal classifier analysis script](https://github.com/mirkomarras/fair-voice/blob/feature/audio_feat_ext/exp_utils/causal_classifier_analysis.py)
executes a counterfactual evaluation by comparing the predictions of the just trained causal classifiers with the predictions
on altered data, where the sensitive attributes (gender, age, language) are flipped to study if the causal classifier is
affected if we hypothesize that a user belonged to a different demographic group. The script can be run as follows:
```
ResNet34
python3 causal_classifier_analysis.py --model /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/classification/far_data__resnet34vox_English-Spanish-train1@15_920_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10/causal_classifier_RF.model --train_set_x /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/classification/train_set_x_resnet34vox.csv --train_set_y /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/classification/train_set_y_resnet34vox.csv

X-Vector
python3 causal_classifier_analysis.py --model /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/classification/far_data__xvector_English-Spanish-train1@13_992_20032022_test_ENG_SPA_75users_6samples_50neg_5pos#00_10/causal_classifier_RF.model --train_set_x /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/classification/train_set_x_xvector.csv --train_set_y /BASE_PATH/dl-fair-voice/exp/counterfactual_fairness/classification/train_set_y_xvector.csv
```

## Paper Results
![audio_features_correlation_before_VIF](https://user-images.githubusercontent.com/26059819/217531998-2b420373-f7a4-4d0d-9390-07e86bd4b784.png)

ResNet34 Random Forest
![ResNet34-RF](https://user-images.githubusercontent.com/26059819/217532038-31d9739e-1552-4a4d-86cd-2ed5c69714ca.png)
![resnet34vox_RF_kde_plot_hue_all](https://user-images.githubusercontent.com/26059819/217532172-be46a700-2ff0-427f-9582-f948742604e3.png)

X-Vector Random Forest
![X-Vector-RF](https://user-images.githubusercontent.com/26059819/217532201-0670f1bf-2def-4c2f-abdb-6f7d1d68345d.png)
![xvector_RF_kde_plot_hue_all](https://user-images.githubusercontent.com/26059819/217532217-32df35e9-dd95-4f3a-889a-a079129c4fa1.png)

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
