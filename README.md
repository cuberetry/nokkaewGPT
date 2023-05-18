# nokkaewGPT
## Install
Project is developed using Python 3.10.8

**Dependencies:**
#### Pytorch

Anaconda: `conda install pytorch`

Pip: `pip install torch`

#### Numpy

`pip install numpy`

#### PyThaiNLP

`pip install pythainlp`

## Setup
To preprocess the corpus before training, simply execute the following command in your terminal or command prompt:
```
$ python preprocess.py
```

To initiate the training process, run the following command in your terminal or command prompt:
```
$ python train.py
```
Before the training begins, the script will prompt you with a question if you wish to reset the model. If you wish to reset the model and start the training from scratch, enter "yes" at the prompt. If you want to continue training with the existing model, simply enter anything else.

After handling the model reset prompt, the script will then ask you to enter the number of training steps you want to execute. This determines the duration of the training process and controls how many iterations the model will go through to optimize its performance. Enter the desired number of training steps and press Enter to proceed with the training.

It's important to configure the number of training steps appropriately based on your specific requirements and available computational resources. Adjusting the number of training steps allows you to balance the training time and the model's learning capacity to achieve the desired results.

To generate output using the trained language model, you can run the following command:
```
$ python generate.py
```
Once the input is provided, the script will generate the output using the trained model and save it to a file named `./output/output_from_model.txt`.

Make sure that you have successfully trained the model before attempting to generate output, as the quality of the generated text relies on the model's training and learned patterns.

## Log
The logs from the train.py script will be stored in the directory `./log/yyyy-mm-dd-hh:mm:ss.txt`, where `yyyy-mm-dd-hh:mm:ss` represents the timestamp of when the script was executed.

Here is an example output from the log file:
```
Ran at 2023-05-17-13:31:19

Reset: False

Hyperparameters:
batch_size: 16
block_size: 32
embedding_dim: 32
hidden_dim: 128
n_embd: 64
n_head: 4
n_layer: 4
dropout: 0.1
vocab_size: 2146

Training steps: 100

0/100
Step 0: Training Loss: 3.3713910579681396, Validation Loss: 3.6702888011932373
100/100
Step 100: Training Loss: 3.3713910579681396, Validation Loss: 3.6702888011932373
```
