# ML-ECG (FSHJAR002 Version)
Welcome to the ML-ECG Platform. We have made it that the code is simple for you to install and get up and running! We use Python virtual environments to run our python and tensorflow code. I have created a make file that will allow you to easily run the exact same experiments as I ran in my paper. Follow the steps below to run the experiments as outlined in my paper.


## Installation
1. The first step to installation is to check the *requirements.txt* file for the correct tensorflow version.  If you are using a Mac computer with an M1 Chip, replace the line which states: "tensorflow" with "tensorflow-macos"
2. Ensure you are in the root "MLECG" directory
3. Run the following commands to set up your virtual environment and install the required dependencies:
```sh
make
```
Your packages should sucessfully be installed. If you wish to remove your virtual environment run the following command
```sh
make clean
```

## Running the Experiments
We will first demonstrate how to run the experiments with predefined makefile commands we have created for you.
We will then show you how to run the program in custom ways.
The main procedures of the codebase happen in the wrapper.py python file.
Please note that you should monitor the running code before model training begins - it may ask for user input.
Training and testing the models under any of the following experiments represents a single round of 10-fold cross validation.
Please see the note on cross-validation at the end of this readme for further information.

### Running the inter-patient classification experiment with proposed model in study
This study's best performing inter-patient classifer developed is a multi-layer LSTM.
To run this study's proposed inter-patient classifier with the optimal hyperparameters found (dropout_rate =0.7, batch_size = 128, epochs=50, num_sec = 0.4 ), type the following:
```sh
make runMain
```
This will create new data splits, so please allow time for this (5-10 minutes if building from scratch).

### Running the intra-patient classification experiment with proposed model in study
To run this study's split-by-beat experiment (i.e. intra-patient classification) type the following:
```sh
make runBeat
```
Note: This will use the same model and hyperparameters as defined in the main inter-patient classification.
This will also create new data splits, so please allow time for this (5-10 minutes if building from scratch).

### Loading already made data splits using proposed model in study
To run with the same model as defined previously, but from previously created data splits, type the following:
```sh
make runLoad:
```
Note 1: This requires data splits to already have been created and saved at:

*"./trained_models/saved_data_splits"*

Note 2: This can be done for either intra- or inter-patient classification type experiments.
		That is, the platform is ambivalent to whether the saved data splits were created
		to respect the patient assignments (i.e. inter-patient classification) or the data was split on
		beats, thus allowing patient-specific data to leak (i.e. intra-patient classification)

## Running the platform in custom ways
Our platform allows you to interact with it in many custom ways. There are multiple arguments that you can pass to the platform for customized model training. 
For help with this you can run the make command:
```sh
make runHelp
```
The platform is run as follows:
```sh
venv/bin/python3 src/wrapper.py "data/" "./trained_models/final_model" <args> <args> <args> .....
```
You may replace the "`<args> <args> <args>...`" with the following arguments to customize model training:

- **-p** : This is used to specify the preprocessing options you require. The options here are "oversample" and "normalize". To normalize and oversample your ECG segments use "-p oversample normalize". Default is None 
- **-e** : How many epochs do you wish to train your model for. Must be >1 and <1024. To run for 500 epochs use argument "-e 500"
- **-i** : Initial Epoch to start running training from. This should only be a different number if training was interupted and needs to start from a checkpoint. Default is zero. Example usage to start training from epoch 10 is: "-i 10"
- **-r** : Flag that can be passed to resume training of a model. The program will look for a backup of a model based on the model path arg provided. There also needs to be saved training and testing data splits.
- **-l** : Specify the learning rate of the model. Example usage to make the model learn with a learning rate of 0.005 is "-l 0.005". Default is 0.001
- **-s** : skip training. Flag that can be passed to skip building and training the model. Thereby moving straight to predictions.It follows then that the model provided with the model path arg is already compiled and trained for at least one epoch. There are also needs to be saved training and testing data splits. To skip training use argument: "-s"
- **-v** : Verbosity. Can choose either -v 1 for verbose output or -v 0 for quiet output. Default is Verbose
- **-b** : The batch size to train your model with. Must be between 1 and 1024. The default is 64. Example usage of specifying a batch size of 32 is "-b 32"
- **-m** : This is the model you can chose to train with. There are many options here. To use the model that found the best success in my study, you can type "-m LSTM_deep3_HighDrop"
- **-d** : Flag that can be passed to disable training with GPU. 
- **-ls** : Each time data splits are created they are saved to numpy files. This argument will load the saved data splits instead of resampling the patients. This is a flag that can be passed to load previously saved training, validation and test data splits from disk.  Note: If resuming training or skipping to predictions, data splits will always be loaded from disk to maintain integrity in the evaulation of the model
- **-n** : Number of seconds to include before and after each R-peak. Default is 0.4. As an example, if you wish to have a 2 second long segment use the argument "-n 1"
- **-x** : Flag that can be passed to perform the beat splitting experiment. This is for creating the intra-patient classifier version of the models. i.e. Patient specific data will be allowed to leak between Training, validation and testing sets

### Deprecated Arguments

- **-t** : Float between 0 and 1 determining how much of the data is to be used for the test split. The remaining is used for training and validation. Example usage of dedicating 20% of the data for the test split is using the argument "-ts 0.2". Default is 0.1

## Cross-Validation Note:

The creation of folds from the data is all automated (i.e. the creation of stratified fold indicies for the training, validation and testing sets).
However, the current approach to 10-fold cross validation required some manual intervention (i.e. selecting a fold number in the *dataptb_xl.py* file).
This was unfortunately due to a memory and bandwidth limitation when training the models through Keras on Google Colab.
Future work would automate this last aspect through making a script that looped through the 10 rounds of cross-validation.

If you would like to create new data splits for a different fold, do the following (in the *dataptb_xl.py* file):
- For patient_splitting, please change the *valid_fold* and *test_fold* variables to different integers between 1 and 10
- If performing beat_splitting, please change the *fold_selected* variable to a different integer between 1 and 10