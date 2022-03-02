# ECEC247_final_project
RNN for EEG signal processing

# Environment Standards
This project is set up using the conda envrionment package manager. To recreate the envrionment the following commands can be run. First we must create the conda environment:

```
conda create --name EEG_RNN python=3.9
```
The required packages can then be installed via the following command:

```
conda activate EEG_RNN
pip install -r requirements.txt
```
At this point you should have a functional python environment identical to the one used in the development of this project. Note, any alternative methods of installing the environment are also acceptable as the `requirements.txt` file is included.

# Pytorch Installation

It should be noted that if you plan to use pytorch to run the project you should install it to fit your system requirements. The command for this can be generated at the [Pytorch Homepage](https://pytorch.org/). For conveinence, if you are running a linux system without GPU, with the `EEG_RNN` envrionment active, you may run:

```
pip3 install torch==1.10.2+cpu torchvision==0.11.3+cpu torchaudio==0.10.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

# Keras Installation

Keras installation is generally much more difficult. This will be generally easier to do exclusively in Google Colabs. 

# EEG Data

As the project spec has requested that we not distribute the data, the `.gitignore` file has been modified to prevent the inclusion of our dataset in the repository. This is additionally helpful, as we do not need to be storing these data sets remotely. 

The current convention being used is unzipping the data into `..\ECEC247_final_project\data\`. Note the unzipped data contains both the MACOSX and windows versions to allow for compatability with anyone's computers. Additionally, following this convention will keep pathing consistent.

