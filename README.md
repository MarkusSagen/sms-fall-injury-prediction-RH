MedBERT Project Regio Halland (RH)
==================================

A repository investigating the use cases of Region Halland in collaboration with
Peltarion and RISE. The project investigates methods for classifying medical fall
injuries and detect when it has been reported as a false negative.

# TODO:
- [ ] Clean up the synthetic dataset notebook
- [ ] Copy notebook and replace / reproduce with synthetic dataset
- [ ] Remove some of the labels for 2016. Not all data should have a label (only 172.250 examples)
- [ ] Make the prediction of test data work
- [ ] Update README on how to test and train BERT
- [-] Clean up the code
- [-] Move the classifiers to one notebook
- [-] **DOUBLE CHECK IF STATED**: Describe BERT and why fails
- [ ] Identify and clarify parts that are assuming that the code is running on Windows TLS servers
- [x] Clean up the BERT code

__Table of Contents__
  - [Project Structure](#project-structure)
  - [Setup](#setup)
  - [Usage](#usage)
    - [Run Lable Corruption on IMDB](#run-label-corruption-on-imdb)
  - [Pushing Commits](#pushing-commits)
    - [Automatic Formating](#automatic-formating)
  - [Work in Progress](#work-in-progress)

## Project Structure
1. medbert - Where the modules for training are stored.
2. scripts - Scripts for executing multiple runs.
3. notebooks -  Jupyter notebooks for testing and plotting results.

## Setup
**NOTE!** Running on Region Halland's Servers does not allow for Docker containers with GPU support!
Therefore, run everythin in a conda environment and the files directly with conda python

Run `make .env` to generate a `.env.tmp` config file based on your configurations in the `.env.template`. **CHECK** that the configurations in that file are as you want them, then run `make build` to build the docker container and `make up` to start the container.

You will need to have docker, make, docker-compose, conda, a NVIDIA GPU, and be able
to run docker without `sudo` in order to run this code. For Mac users, make sure to
have make version 4+. It can be installed easily with `brew install make` and then
using the `gmake` binary instead of `make`.

## Usage
### Run Label Corruption on IMDB

  1. Ensure that you have copied the `.env.template` file to `.env` and filled in the forms.
  2. Run `make build && make up` to initialize the Docker image
  3. Run the training script
  ```bash
  make run cmd="medbert/label_recon.py \
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
  },
  ```
  4. Run full suite of jobs with different fractions of destroyed positive labels - with and without label smoothing:
  ```bash
  bash scripts/label_recon_suite.sh
  bash scripts/label_recon_suite_label_smoothing.sh
  ```
  5. Plot the results by running the notebook [here](https://github.com/Peltarion/medbert-label-reconstruction/blob/master/notebooks/Explore%20Incorrect%20Predictions.ipynb)
  6. When done terminate the Docker image: `make down`

##### Explaining the Arguments
- `--dataset` = {imdb, rh, mimic-iii}, where rh is Region Halland's dataset when running the script on their servers
- `--loggers` = {wandb, tensorboard, all, None}
- `--num_train_samples` - How many training samples to use. 50/50 of each label (0s and 1s)
- `--corrupt_percentage` - [0, 100] Percentage of how many of the positive labels (1) to change to 0s
- `--label_smoothing` - [0.0, 1.0] How much label smoothing to apply. 0.0 defaults to regular one-hot encoded CategoricalCrossEntropy Loss.

##### Example:
If --num_train_samples=10000 and --corrupt_percentage=80,
then 4000 of the 5000 positive labels will have their labels changed to zeros.

## Pushing Commits
Before pushing new commits make sure to format the code.
This repo uses `black` for its formating and `tox`.
Verify all code checks before sending up a pull request with
`make run cmd="tox"`. The formating configs are defined in `setup.cfg` and `tox.ini`.

### Automatic formating
If you want to format a specific file you can run:
`make run cmd="black medbert/label_recon.py"`
or install `black` locally and run it from there.

## Work in Progress
See the [Projects Tab](https://github.com/Peltarion/medbert-label-reconstruction/projects)
to see what features are considered, worked on and current status.

## notes
- addded "dataset_path" to RegionHallandDatasetModule
