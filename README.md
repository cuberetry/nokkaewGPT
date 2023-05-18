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

**Dataset:**

In addition to the data included in the Git repository, it is highly recommended to incorporate the dataset `thaisum.csv` from the following GitHub repository: https://github.com/nakhunchumpolsathien/ThaiSum.

By including this dataset, you will be able to leverage a comprehensive and diverse set of Thai text data, enhancing the effectiveness and accuracy of the model's performance. It is advised to download and incorporate the theism.csv dataset into your project to ensure optimal results and compatibility with the established production environment.

Furthermore, please ensure that the dataset file `thaisum.csv` is placed in the designated directory within your project structure. Specifically, it is recommended to store the `thaisum.csv` file in the `./data/` directory.

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

## Model Configuration
The performance and behavior of the NokkaewGPT model are influenced by various hyperparameters that govern its architecture and training process. These hyperparameters determine the model's capacity, memory usage, and computational efficiency.

1. Batch Size
The batch size refers to the number of training examples processed in each iteration of the training algorithm. Adjusting the batch size can impact training speed and memory requirements.

2. Block Size
The block size is a parameter that determines the length of the input sequences processed by the model. The block size affects the maximum sequence length that the model can handle efficiently.

3. Number of Embedding Dimensions (n_embd)
The number of embedding dimensions, represented by n_embd, specifies the size of the embedding vectors used to represent input tokens. Larger n_embd values can allow the model to capture more intricate patterns and semantic information.

4. Number of Attention Heads (n_head)
The number of attention heads, denoted as n_head, determines the level of attention and parallelism in the model's self-attention mechanism. Increasing the number of attention heads can enhance the model's ability to capture different dependencies and relationships within the input.

5. Number of Layers (n_layer)
The number of layers, represented by n_layer, indicates the depth of the model's architecture. Deeper models can capture more complex patterns and hierarchies in the data, but they may require additional computational resources.

The table below presents a set of recommended hyperparameters for users to experiment with when using the NokkaewGPT model.

| Hyperparameter | S    | M1    | M2    | L   |
| :---: | :---: | :---: | :---: | :---: |
| batch_size | 32   | 64   | 32   | 64   |
| block_size | 1024 | 1024 | 2048 | 2048 |
| n_embd | 768  | 768  | 1024 | 1024 |
| n_head | 12   | 12   | 12   | 12   |
| n_layer | 24   | 24   | 24   | 24   |
