# NLP From Scratch: Translation with a Sequence to Sequence Network and Attention

This repository implements a neural network that learns to translate sentences from French to English using a Sequence-to-Sequence (Seq2Seq) model and an attention mechanism. This is based on Sean Robertson's tutorial.

## Overview

In this project, we build a neural network that translates from French to English. The model utilizes the Sequence-to-Sequence (Seq2Seq) architecture, where two Recurrent Neural Networks (RNNs) work together—an encoder and a decoder. The encoder compresses the input sequence into a vector, and the decoder unfolds this vector into a translated output sequence.

To enhance the model's performance, we implement an attention mechanism, allowing the decoder to focus on specific parts of the input sequence during translation.

## Project Structure

1. **Data Processing:**
   - The dataset consists of English-French translation pairs.
   - The data is cleaned, normalized, and split into training pairs.
   
2. **Model Architecture:**
   - **EncoderRNN**: Encodes the input sequence into a hidden state.
   - **DecoderRNN**: Decodes the hidden state into a sequence of output tokens.
   - **BahdanauAttention**: Implements the attention mechanism.
   - **AttnDecoderRNN**: Combines the attention mechanism with the decoder.

3. **Training**: 
   - The model is trained with a loss function (Negative Log Likelihood Loss) and an optimizer (Adam).
   - We use Teacher Forcing during training for better convergence.

4. **Evaluation**: 
   - After training, the model is evaluated by generating translations for randomly selected sentences.

## Setup

### Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib (for plotting results)

Install dependencies using pip:

```bash
pip install torch numpy matplotlib
```

### Data

Download the data from [here](http://www.manythings.org/anki/) and extract it to the current directory. The data consists of English-French sentence pairs.

### Preparing Data

To prepare the data, we use functions to read the translation pairs, normalize the sentences, and filter them by length. The maximum sentence length is limited to 10 words to speed up training.

```python
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
```

### Model Definition

The model consists of the following components:

#### Encoder

```python
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        # Initializes the encoder network with embedding and GRU layers
```

#### Decoder

```python
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        # Initializes the decoder network with embedding, GRU, and output layers
```

#### Attention Mechanism

```python
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        # Defines the attention mechanism
```

#### Attention-Based Decoder

```python
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        # Combines the attention mechanism with the decoder
```

## Training the Model

```python
def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001):
    # Train the model for n_epochs
```

During training, we use the Adam optimizer and the Negative Log Likelihood loss function to update the model's parameters.

## Evaluation

```python
def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    # Evaluate the trained model on a sentence
```

The evaluation function generates translations by feeding the encoder's outputs into the decoder and using the decoder's predictions to translate a given sentence.

## Plotting Training Losses

```python
import matplotlib.pyplot as plt

def showPlot(points):
    # Plot the training loss curve
```

## Usage

1. **Train the Model**

   Start training by calling the `train()` function with the prepared dataloader and model components:

   ```python
   train(train_dataloader, encoder, decoder, n_epochs=10)
   ```

2. **Evaluate Translations**

   Evaluate the model's translations by passing sentences to the `evaluate()` function:

   ```python
   output_words, decoder_attn = evaluate(encoder, decoder, "je suis tres mediocre en sport", input_lang, output_lang)
   print(" ".join(output_words))
   ```

## Example

For the sentence "je suis tres mediocre en sport", the model should output something like:

```text
> je suis tres mediocre en sport
= i am very poor at sports
< i am very poor at sports
```

Feel free to open issues if you encounter any problems or have questions about the implementation!
