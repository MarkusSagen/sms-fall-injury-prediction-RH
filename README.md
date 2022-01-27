SMS Project Region of Halland (RH)
==================================

Repository of code used in the Vinnova funded application for "Svenskt Medicinskt Språkdatalabb". One of the use-cases in the project was identifying previous fall injuries that occured for patients during their hospital visits, from the patient's journal text. The goal of the project was to verify if machine learning models could learn to identify which patients had a fall injury in the past, and (if possible), extend it to predict which patients would be in the risk for suffering from a future fall injury.

This repository is code for one of the use-cases in the SMS project, and is a collaboration between Region Halland, Peltarion and RISE. For code used for Region Västra Götaland Folktandvården and explainable antibiotic predictions, see [here](https://github.com/Peltarion/sms-explainable-antibiotics-VGR).

The repository contains code for reporducing our results in the form of Juptyter Notebooks ([notebooks](./notebooks)) and code for training a Swedish BERT model on medical data ([medbert](./medbert)). The notebooks are structured in such a way to show the process we used for accessing and processing the data; training and comparing the different models for identifying past fall injuries, and an initial notebook for predicting future fall injuries.

# About the notebooks and reproducability
The notebooks have been structured in such a way as to be illustrative of the methodology used for pre-processing and training the models on the real dataset from Region Halland. Because of the senisitive nature of the dataset, the original data can not be shared. But, to still priovide insight and reproducability, we have created a dataset to reflect certain features in the original dataset. The journal text from patients have been replaced entirely with positive or negative review from the Amazon book review dataset. This is because we still wanted a free text dataset with signal for calssification. Instead trying to anonymize the real data, which we can not garantee would comply fully with [GDPR](https://gdpr-info.eu/) and [patientskyddsdatalagen](https://www.riksdagen.se/sv/dokument-lagar/dokument/svensk-forfattningssamling/patientsakerhetslag-2010659_sfs-2010-659). This way we could garantee privacy and safety, while providing reproducable results.

The new dataset is created in such a way to resemble known features of the original dataset, such as the distribution of patient age, gender, number of patients admitted during a year, average hospital duration, patients with annotated (known) fall injuries or not, etc.

---

## Table of Contents
  - [Project Structure](#project-structure)
  - [Technologies](#technologies)
  - [Setup](#setup)
  - [Usage](#usage)
  - [Extra](#extra)
    - [Test Lable Corruption on IMDB](#test-if-label-corruption-can-be-identified-from-softmax-values)


## Project Structure
1. notebooks -  Jupyter notebooks for illustrating most of the metods used and why.
2. medbert - Where the modules for training the Swedish BERT model are stored.
3. scripts - Scripts for training the BERT models.

## Technologies
The following code and notebooks are written to work both in a Windows and Linux Environment, and does not require a dedicated GPU - unless one wishes to train the BERT model.

The project requires the following:
- Python 3.6+ and Jupyter Notebook
- We strongly recommend using a virtual environment solution for installing and running the packages, such as Anaconda, Pyenv or pipenv.

## Setup
1. Activate your virtual environment of choice
2. ```pip install -r requirements.txt```

## Usage
For using the notebooks:
1. ```jupyter notebook```

To run the Swedish BERT model for training:
1. ```chmod -R 777 scripts```
2. ```bash scripts/run_command.sh```

### Extra: Test if label corruption can be identified from Softmax values.
<details>
  <summary>Testing on the IMDB classification dataset.</summary>

  1. Run the training script
  ```bash
  make run cmd="medbert/run.py \
    --dataset=imdb                     \
    --loggers=all                      \
    --model_name=bert-base-uncased     \
    --max_sequence_length=512          \
    --label_smoothing=0.0              \
    --num_train_samples=10000          \
    --corrupt_percentage=0             \
    --fp16                             \
    --seed=42"
  ```
  To remotely debug the script with vscode, use `make debug cmd=` instead, and
  add the following to the `launch.json` file:
  ```json
  {
      "name": "docker debug",
      "type": "python",
      "request": "attach",
      "justMyCode": false,
      "host": "127.0.0.1",
      "port": 5678,
      "pathMappings": [
          {
              "localRoot": "${workspaceFolder}",
              "remoteRoot": "/workspace"
          }
      ]
  }
  ```

#### Explaining the Arguments
- `--dataset` = {imdb, rh, mimic-iii}, where rh is Region Halland's dataset when running the script on their servers
- `--loggers` = {wandb, tensorboard, all, None}
- `--num_train_samples` - How many training samples to use. 50/50 of each label (0s and 1s)
- `--corrupt_percentage` - [0, 100] Percentage of how many of the positive labels (1) to change to 0s
- `--label_smoothing` - [0.0, 1.0] How much label smoothing to apply. 0.0 defaults to regular one-hot encoded CategoricalCrossEntropy Loss.

</details>


