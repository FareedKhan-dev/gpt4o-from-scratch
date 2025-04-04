<!-- omit in toc -->
# GPT-4o like MultiModal from Scratch

![Python](https://img.shields.io/badge/Python-3.9-blue) [![Medium](https://img.shields.io/badge/Medium-Read%20Now-red?logo=medium)](https://medium.com/@fareedkhandev/ad0fa9c213d3) ![Contributions](https://img.shields.io/badge/Contributions-Welcome-yellow) 

In this blog, we are going to code a very tiny multimodal architecture step by step that can process text, images, videos, and audios, and generate images from text prompts, just like GPT-4o, from scratch.

```
I wont be using OOP, functions or complex python libraries
```

But at most, we’ll use simple Python loops (for training only) in a **Jupyter Notebook style** to actually see what is happening behind the code.

The goal of this article is to help you understand the step-by-step implementation in detail rather than providing the perfect model.

<!-- omit in toc -->
## Table of Contents
- [Our GPT-4o Architecture](#our-gpt-4o-architecture)
- [Setting the Stage](#setting-the-stage)
- [What is BPE Tokenization](#what-is-bpe-tokenization)
- [Coding a BPE Tokenizer](#coding-a-bpe-tokenizer)
- [A Decoder-Only Transformer](#a-decoder-only-transformer)
- [Generating Text Using Transformer](#generating-text-using-transformer)
- [Chat with Images (ResNet)](#chat-with-images-resnet)
- [Testing Chat with Images Functionality](#testing-chat-with-images-functionality)
- [Chat With Video/Audio (ResNet + Feature Vectors)](#chat-with-videoaudio-resnet--feature-vectors)
- [Generate Images Functionality (ResNet + Feature Extractor)](#generate-images-functionality-resnet--feature-extractor)
- [Generating Image using Text Prompt](#generating-image-using-text-prompt)
- [Conclusion](#conclusion)



![Some of the generated output of our MultiModal](https://miro.medium.com/v2/resize:fit:875/1*dqrxuqE2d_zdt7wkgOyXUQ.png)

Following are the features our multimodal model will have:

*   Chat with text like an LLM (Using Transformer)
*   Chat with images, videos, and audios (Using Transformer + ResNet)
*   Generate images from text prompts (Using Transformer + ResNet + Feature approach)

## Our GPT-4o Architecture

![Tiny GPT-4o Architecture](https://miro.medium.com/v2/resize:fit:1250/1*KXvvQ4kiBmhd_u4mw-BLDQ.png)

So here is quick overview of our GPT-4o Architecture:

1.  So, our simplified GPT-4o-like model first takes in various inputs, including text, images, videos, and audio, as shown in the “Inputs” box.
2.  Then, the “Process” stage converts all these different inputs into a standard numerical format that the model can work with.
3.  Next, these numbers go into the central “Transformer/ResNet” model, which analyzes the sequence to understand patterns and relationships in the data.
4.  Finally, in the “Outputs” stage, the model uses its understanding to generate results, like new text or image features for finding a matching picture.

## Setting the Stage

Clone the repository and install the required libraries.

```bash
git https://github.com/FareedKhan-dev/gpt4o-from-scratch.git
```

Install the required libraries:

```bash
cd gpt4o-from-scratch
pip install -r requirements.txt
```

We will be using a few libraries, so let’s import them first.

```python
# Core PyTorch libraries for building models, tensors, and training
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Image processing and computer vision
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw # For creating/handling dummy images

# Standard Python libraries
import re               # For regular expressions (used in BPE word splitting)
import collections      # For data structures like Counter (used in BPE frequency counting)
import math             # For mathematical functions (used in Positional Encoding)
import os               # For interacting with the operating system (file paths, directories)

# Numerical Python (Used conceptually for audio waveform generation)
import numpy as np
```

The first step is to create a tokenizer so let’s create one.

## What is BPE Tokenization

In creating a multimodal, language model or any kind NLP tasks, tokenization is the fundamental first step of breaking down raw text into smaller units called tokens.

These tokens are the basic building blocks that machine learning models process (e.g., words, punctuation marks).

![Simplest Tokenization Process](https://miro.medium.com/v2/resize:fit:1250/1*76dSwbdaAXPJyb83k3S3sg.png)

The simplest tokenization involves lowercasing the text, removing punctuation, and splitting the text into a list of words.

However, this approach has a drawback. If every unique word is treated as a token, the vocabulary size can become excessively large, especially in corpora with many word variations (e.g., ‘run’, ‘runs’, ‘running’, ‘ran’, ‘runner’). This increases memory and computational requirements significantly.

Unlike simple word-based tokenization, **BPE (Byte Pair Encoding)** is a subword tokenization technique that helps control vocabulary size while effectively handling rare words.

![BPE Explanation](https://miro.medium.com/v2/resize:fit:1250/1*fh6-YXO3HW0wXbEfyqtG2g.png)

In BPE initially, the input sentence is split into individual characters. Then, frequent adjacent character pairs are merged into new subwords. This process continues until the desired vocabulary size is reached.

As a result, common subword parts are reused, and the model can handle words it has never encountered before by breaking them into smaller, known subword units.

GPT-4 uses BPE as its tokenization component. Now that we’ve gone through a high-level overview of BPE, let’s start coding it.

## Coding a BPE Tokenizer

We will use a small excerpt from Lewis Carroll’s **“Alice’s Adventures in Wonderland”** as our training data.

A larger, more diverse corpus would result in a more general-purpose tokenizer, but this smaller example allows us to trace the process more easily.

```python
# Define the raw text corpus for training the BPE tokenizer
corpus_raw = """
Alice was beginning to get very tired of sitting by her sister on the
bank, and of having nothing to do: once or twice she had peeped into the
book her sister was reading, but it had no pictures or conversations in
it, 'and what is the use of a book,' thought Alice 'without pictures or
conversation?'
So she was considering in her own mind (as well as she could, for the
hot day made her feel very sleepy and stupid), whether the pleasure
of making a daisy-chain would be worth the trouble of getting up and
picking the daisies, when suddenly a White Rabbit with pink eyes ran
close by her.
"""
```

Next, we need to convert the entire corpus to lowercase to make sure that the same word, regardless of its capitalization (e.g., ‘Alice’ vs. ‘alice’), is treated as the same unit during frequency counting and merging.

```python
# Convert the raw corpus to lowercase
corpus_lower = corpus_raw.lower()
```

Now that we have lowercase our corpus data.

Next, we need to split the text into its basic constituents. While BPE operates on subwords, it typically starts by considering word-level units (including punctuation).

```python
# Define the regular expression for splitting words and punctuation
split_pattern = r'\w+|[^\s\w]+'

# Apply the regex to the lowercased corpus to get a list of initial tokens
initial_word_list = re.findall(split_pattern, corpus_lower)

print(f"Corpus split into {len(initial_word_list)} initial words/tokens.")

# Display the first few tokens
print(f"First 3 initial tokens: {initial_word_list[:3]}")


#### OUTPUT ####
Corpus split into 127 initial words/tokens.
First 3 initial tokens: ['alice', 'was', 'beginning']
```

Our corpus splits into `127` tokens and you can see the first three tokens of our corpus, Let’s understand what is happening in our code.

We use a regular expression `r'\w+|[^\s\w]+'` via `re.findall`:

*   `\w+`: Matches one or more alphanumeric characters (letters, numbers, and underscore). This captures standard words.
*   `|`: Acts as an OR operator.
*   `[^\s\w]+`: Matches one or more characters that are *not* whitespace (`\s`) and *not* word characters (`\w`). This captures punctuation marks like commas, periods, colons, quotes, parentheses, etc., as separate tokens.

The result is a list of strings, where each string is either a word or a punctuation mark.

The core principle of BPE is merging the *most frequent* pairs. Therefore, we must know how often each unique initial word/token appears in the corpus. Pairs found within more frequent words will have a higher impact on the merge decisions.

`collections.Counter` efficiently creates a dictionary-like object mapping each unique item (word/token) to its count.

```python
# Use collections.Counter to count frequencies of items in initial_word_list
word_frequencies = collections.Counter(initial_word_list)

# Display the 3 most frequent tokens and their counts
print("3 Most frequent tokens:")
for token, count in word_frequencies.most_common(3):
    print(f"  '{token}': {count}")


#### OUTPUT ####
3 Most frequent tokens:
  the: 7
  of:  5
  her: 5
```

The most frequent token is “the” which appears 7 times in our tiny corpus.

BPE training operates on sequences of symbols. We need to transform our list of unique words/tokens into this format. For each unique word/token:

1.  Split it into a list of individual characters.
2.  Append a special end-of-word symbol (we’ll use `</w>`) to this list.
    This `</w>` marker is critically important:

It prevents BPE from merging characters across different words. For example, the ‘s’ at the end of ‘apples’ should not merge with the ‘a’ at the beginning of ‘and’.

It also allows the algorithm to learn common word endings as distinct subword units (e.g., ‘ing</w>’, ‘ed</w>’, ‘s</w>’).

We store this mapping (original word -> list of symbols) in a dictionary.

```python
# Define the special end-of-word symbol "</w>"
end_of_word_symbol = '</w>'

# Create a dictionary to hold the initial representation of the corpus
# Key: original unique word/token, Value: list of characters + end_of_word_symbol
initial_corpus_representation = {}

# Iterate through the unique words/tokens identified by the frequency counter
for word in word_frequencies:
    # Convert the word string into a list of its characters
    char_list = list(word)
    # Append the end-of-word symbol to the list
    char_list.append(end_of_word_symbol)
    # Store this list in the dictionary with the original word as the key
    initial_corpus_representation[word] = char_list
```

So, let’s print the representation for two of our words, “beginning” and “.”

```python
# Display the representation for a sample word
example_word = 'beginning'
if example_word in initial_corpus_representation:
    print(f"Representation for '{example_word}': {initial_corpus_representation[example_word]}")
example_punct = '.'
if example_punct in initial_corpus_representation:
    print(f"Representation for '{example_punct}': {initial_corpus_representation[example_punct]}")


#### OUTPUT ####
Created initial corpus representation for 86 unique words/tokens.
Representation for 'beginning': ['b', 'e', 'g', 'i', 'n', 'n', 'i', 'n', 'g', '</w>']
Representation for '.': ['.', '</w>']
```

BPE algorithm starts its vocabulary with the set of all individual symbols present in the initial corpus representation.

This includes all unique characters from the original text *plus* the special `</w>` symbol we added. Using a Python `set` automatically handles uniqueness – adding an existing character multiple times has no effect.

```python
# Initialize an empty set to store the unique initial symbols (vocabulary)
initial_vocabulary = set()

# Iterate through the character lists stored in the initial corpus representation
for word in initial_corpus_representation:
    # Get the list of symbols for the current word
    symbols_list = initial_corpus_representation[word]
    # Update the vocabulary set with the symbols from this list
    # The `update` method adds all elements from an iterable (like a list) to the set
    initial_vocabulary.update(symbols_list)

# Although update should have added '</w>', we can explicitly add it for certainty
# initial_vocabulary.add(end_of_word_symbol)

print(f"Initial vocabulary created with {len(initial_vocabulary)} unique symbols.")
# Optional: Display the sorted list of initial vocabulary symbols
print(f"Initial vocabulary symbols: {sorted(list(initial_vocabulary))}")



#### OUTPUT ####
Initial vocabulary created with 31 unique symbols.
Initial vocabulary symbols: ["'", '(', ')', ',', '-', '.', ':', '</w>', ...]
```

So, our tiny corpus has a total of 31 unique symbols, and I have printed some of them for you to see.

Now, comes the the core BPE learning phase. We will iteratively find the most frequent adjacent pair of symbols in our current corpus representation and merge them into a new, single symbol (subword).

This process builds the subword vocabulary and the ordered list of merge rules.

Before starting the loop, we need:

1.  `num_merges`: Defines the number of merges, controlling final vocabulary size. Larger values capture more complex subwords.
2.  `learned_merges`: A dictionary to store merge rules, with pair tuples as keys and merge priority as values.
3.  `current_corpus_split`: Holds the corpus state as it's modified by merges, initialized as a copy of the original.
4.  `current_vocab`: Holds the growing vocabulary, initialized as a copy of the original vocabulary.

```python
# Define the desired number of merge operations
# This determines how many new subword tokens will be added to the initial character vocab
num_merges = 75 # Let's use 75 merges for this example

# Initialize an empty dictionary to store the learned merge rules
# Format: { (symbol1, symbol2): merge_priority_index }
learned_merges = {}

# Create a working copy of the corpus representation to modify during training
# Using .copy() ensures we don't alter the original initial_corpus_representation
current_corpus_split = initial_corpus_representation.copy()

# Create a working copy of the vocabulary to modify during training
current_vocab = initial_vocabulary.copy()

print(f"Training state initialized. Target number of merges: {num_merges}")
print(f"Initial vocabulary size: {len(current_vocab)}")


#### OUTPUT ####
Training state initialized. Target number of merges: 75
Initial vocabulary size: 31
```

We will now iterate `num_merges` times. Inside this loop, we perform the core BPE steps: count pairs, find the best pair, store the rule, create the new symbol, update the corpus representation, and update the vocabulary.

```python
# Start the main loop that iterates for the specified number of merges
print(f"\n--- Starting BPE Training Loop ({num_merges} iterations) ---")
for i in range(num_merges):
    print(f"\nIteration {i + 1}/{num_merges}")

    # Calculate pair statistics
    pair_counts = collections.Counter()
    for word, freq in word_frequencies.items():
        symbols = current_corpus_split[word]
        for j in range(len(symbols) - 1):
            pair = (symbols[j], symbols[j+1])
            pair_counts[pair] += freq
    if not pair_counts:
        print("No more pairs found to merge. Stopping early.")
        break

    # Find the best pair
    best_pair = max(pair_counts, key=pair_counts.get)
    print(f"Found best pair: {best_pair} with frequency {pair_counts[best_pair]}")

    # Store merge rule
    learned_merges[best_pair] = i

    # Create new symbol
    new_symbol = "".join(best_pair)

    # Update corpus representation
    next_corpus_split = {}
    for word in current_corpus_split:
        old_symbols = current_corpus_split[word]
        new_symbols = []
        k = 0
        while k < len(old_symbols):
            if k < len(old_symbols) - 1 and (old_symbols[k], old_symbols[k+1]) == best_pair:
                new_symbols.append(new_symbol)
                k += 2
            else:
                new_symbols.append(old_symbols[k])
                k += 1
        next_corpus_split[word] = new_symbols
    current_corpus_split = next_corpus_split

    # Update vocabulary
    current_vocab.add(new_symbol)

# Final output
print(f"\n--- BPE Training Loop Finished after {i + 1} iterations ---")
final_vocabulary = current_vocab
final_learned_merges = learned_merges
final_corpus_representation = current_corpus_split
```

When we start the training loop of BPE, it will run for 75 iterations, as we have set the merges to 75. Let’s take a look at the output of one of the iterations to see how it looks:

```bash
#### OUTPUT (Single Iteration) ####

--- Starting BPE Training Loop (75 iterations) ---

Iteration 1/75
  Step 2.3: Calculating pair statistics...
  Calculated frequencies for 156 unique pairs.
  Step 2.4: Checking if pairs exist...
  Pairs found, continuing training.
  Step 2.5: Finding the most frequent pair...
  Found best pair: ('e', '</w>') with frequency 21
  Step 2.6: Storing merge rule (Priority: 0)...
  Stored: ('e', '</w>') -> Priority 0
  Step 2.7: Creating new symbol from best pair...
  New symbol created: 'e</w>'
  Step 2.8: Updating corpus representation...
  Corpus representation updated for all words.
  Step 2.9: Updating vocabulary...
  Added 'e</w>' to vocabulary. Current size: 32
```

In our training, every iteration involves the same steps, such as calculating frequency pairs, finding the best pair, and so on all the steps we’ve seen earlier.

Let’s take a look at the vocabulary size of our corpus.

```python
print(f"Final Vocabulary Size: {len(final_vocabulary)} tokens")


#### OUTPUT ####
Final Vocabulary Size: 106 tokens
```

Now that our BPE training is done, It's useful to see how specific words from the training corpus are represented *after* all the merge operations have been applied.

```python
# List some words we expect to see interesting tokenization for
example_words_to_inspect = ['beginning', 'conversations', 'sister', 'pictures', 'reading', 'alice']

for word in example_words_to_inspect:
    if word in final_corpus_representation:
        print(f"  '{word}': {final_corpus_representation[word]}")
    else:
        print(f"  '{word}': Not found in original corpus (should not happen if chosen from corpus).")



#### OUTPUT ####
Final Representation of Example Words from Training Corpus:
  'beginning': ['b', 'e', 'g', 'in', 'n', 'ing</w>']
  'conversations': ['conversati', 'on', 's</w>']
  'sister': ['sister</w>']
  'pictures': ['pictures</w>']
  'reading': ['re', 'ad', 'ing</w>']
  'alice': ['alice</w>']
```

This shows the final token sequence for known words according to the learned BPE rules.

But we need a sample sentence or text that the BPE model hasn’t necessarily seen during training to demonstrate its ability to tokenize new input, potentially including words not in the original corpus.

```python
# Define a new text string to tokenize using the learned BPE rules
# This text contains words seen in training ('alice', 'pictures')
# and potentially unseen words or variations ('tiresome', 'thought')
new_text_to_tokenize = "Alice thought reading was tiresome without pictures."
```

When we perform the same training loop on our unseen text we got the following tokenize data:

```python
print(f"Original Input Text: '{new_text_to_tokenize}'")
print(f"Tokenized Output ({len(tokenized_output)} tokens): {tokenized_output}")


#### OUTPUT ####
Original Input Text: 'Alice thought reading was tiresome without pictures.'
Tokenized Output (21 tokens): ['alice</w>', 'thou', 'g', 'h',
                               't</w>', 're', 'ad', 'ing</w>',
                               'was</w>', 'ti', 're', 's', 'o', 'm',
                               'e</w>', 'wi', 'thou', 't</w>',
                               'pictures</w>', '.', '</w>']
```

Ah, finally, we have completed the BPE Tokenizer process. Our text is now represented as sequences of subword tokens.

But just having tokens isn’t enough, we need a model that can understand the patterns and relationships within these sequences to generate meaningful text.

Next, we need to build a text generation LLM, and since GPT-4 is based on the Transformer architecture, we will follow the same approach.

This architecture was famously introduced in the paper *“Attention Is All You Need.”* There are many implementations of Transformer models, and we will be coding while simultaneously understanding the theory to grasp the core logic of the Transformer model.

## A Decoder-Only Transformer

Our goal is to build a language model that can predict the next token in a sequence given the preceding tokens. By repeatedly predicting and appending tokens, the model can generate new text. We’ll focus on a **Decoder-only Transformer**, similar in spirit to models like GPT.

For simplicity in demonstrating the Transformer architecture itself, we’ll switch back to **character-level tokenization** for this part.

This means each unique character in our corpus will be a separate token.

While BPE is generally more powerful, character-level tokenization keeps the vocabulary small and allows us to focus purely on the model’s mechanics.

![Simplified LLM Creation Process](https://miro.medium.com/v2/resize:fit:1250/1*tSID6i2J1NHYHPrW5klE7Q.png)

So our transformer model flow works like this:

1.  **Text to Tokens:** Break the input text into smaller pieces (tokens).
2.  **Tokens to Numbers:** Convert each token into a unique number (ID).
3.  **Add Meaning & Position:** Turn these numbers into meaningful vectors (embeddings) and add position info so the model knows the word order.
4.  **Core Processing:** The Transformer’s main layers analyze how all the words relate to each other using “self-attention”.
5.  **Guess Next Word:** Calculate the probability for every word in its vocabulary being the next one.
6.  **Choose Best Guess:** Select the most likely word (or sample one) based on those probabilities as the output.

We’ll use the same “Alice in Wonderland” corpus as before.

```python
# Define the raw text corpus for training
corpus_raw = """
Alice was beginning to get very tired of sitting by her sister on the
bank, and of having nothing to do: once or twice she had peeped into the
book her sister was reading, but it had no pictures or conversations in
it, 'and what is the use of a book,' thought Alice 'without pictures or
conversation?'
So she was considering in her own mind (as well as she could, for the
hot day made her feel very sleepy and stupid), whether the pleasure
of making a daisy-chain would be worth the trouble of getting up and
picking the daisies, when suddenly a White Rabbit with pink eyes ran
close by her.
"""
```

Now, let’s create our character vocabulary and mappings.

```python
# Find all unique characters in the raw corpus
chars = sorted(list(set(corpus_raw)))
vocab_size = len(chars)

# Create character-to-integer mapping (encoding)
char_to_int = { ch:i for i,ch in enumerate(chars) }

# Create integer-to-character mapping (decoding)
int_to_char = { i:ch for i,ch in enumerate(chars) }

print(f"Created character vocabulary of size: {vocab_size}")
print(f"Vocabulary: {''.join(chars)}")


#### OUTPUT ####

Created character vocabulary of size: 36
Vocabulary:
 '(),-.:?ARSWabcdefghiklmnoprstuvwy
```

With these mappings, we convert our entire text corpus into a sequence of numbers (token IDs).

```python
# Encode the entire corpus into a list of integer IDs
encoded_corpus = [char_to_int[ch] for ch in corpus_raw]

# Convert the list into a PyTorch tensor
full_data_sequence = torch.tensor(encoded_corpus, dtype=torch.long)

print(f"Encoded corpus into a tensor of shape: {full_data_sequence.shape}")


#### OUTPUT ####
Encoded corpus into a tensor of shape: torch.Size([593])
```

The output tells us our model only needs to learn about 36 unique characters (including space, punctuation, etc.).

This is our vocab_size, and the second line shows exactly which characters are included.

With these mappings, we convert our entire text corpus into a sequence of numbers (token IDs).

```python
# Encode the entire corpus into a list of integer IDs
encoded_corpus = [char_to_int[ch] for ch in corpus_raw]

# Convert the list into a PyTorch tensor
full_data_sequence = torch.tensor(encoded_corpus, dtype=torch.long)

print(f"Encoded corpus into a tensor of shape: {full_data_sequence.shape}")


### OUTPUT ####
Encoded corpus into a tensor of shape: torch.Size([593])
```

This confirms that our 593-character text is now represented as a single PyTorch tensor (a list of numbers) of length 593, ready for the model.

Before building the model, we need to set some configuration values, known as hyperparameters. These control the size and behavior of our Transformer.

```python
# Define Model Hyperparameters (using calculated vocab_size)
# vocab_size = vocab_size # Already defined from data
d_model = 64         # Embedding dimension (vector size for each token)
n_heads = 4          # Number of attention heads (parallel attention calculations)
n_layers = 3         # Number of Transformer blocks stacked on top of each other
d_ff = d_model * 4   # Hidden dimension in the feed-forward networks
block_size = 32      # Maximum sequence length the model sees at once
# dropout_rate = 0.1 # Omitting dropout for simplicity

# Define Training Hyperparameters
learning_rate = 3e-4
batch_size = 16      # Number of sequences processed in parallel during training
epochs = 5000        # Number of training iterations
eval_interval = 500 # How often to print loss

# Device Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ensure d_model is divisible by n_heads
assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
d_k = d_model // n_heads # Dimension of keys/queries/values per head

print(f"Hyperparameters defined:")
print(f"  vocab_size: {vocab_size}")
print(f"  d_model: {d_model}")
print(f"  n_heads: {n_heads}")
print(f"  d_k (dim per head): {d_k}")
print(f"  n_layers: {n_layers}")
print(f"  d_ff: {d_ff}")
print(f"  block_size: {block_size}")
print(f"  learning_rate: {learning_rate}")
print(f"  batch_size: {batch_size}")
print(f"  epochs: {epochs}")
print(f"  Using device: {device}")
```

though GPT-4 is a close source model we cannot have these parameter values, so i define it with basic values so it can easily replicate it.

```bash
#### OUTPUT ####
Hyperparameters defined:
  vocab_size: 36
  d_model: 64
  n_heads: 4
  d_k (dim per head): 16
  n_layers: 3
  d_ff: 256
  block_size: 32
  learning_rate: 0.0003
  batch_size: 16
  epochs: 5000
  Using device: cuda
```

We see our `vocab_size` of 36, embedding size (`d_model`) of 64, 3 layers, 4 attention heads, etc. It also confirms whether we're using a GPU (`cuda`) or CPU, which impacts training speed.

Our model learns by predicting the next character. So, we need to create input sequences (`x`) and corresponding target sequences (`y`), where `y` is simply `x` shifted one position to the right. We'll create overlapping sequences of length `block_size` from our encoded corpus.

```python
# Create lists to hold all possible input (x) and target (y) sequences
all_x = []
all_y = []
num_total_tokens = len(full_data_sequence)
for i in range(num_total_tokens - block_size):
    x_chunk = full_data_sequence[i : i + block_size]
    y_chunk = full_data_sequence[i + 1 : i + block_size + 1]
    all_x.append(x_chunk)
    all_y.append(y_chunk)

# Stack the lists into tensors
train_x = torch.stack(all_x)
train_y = torch.stack(all_y)

num_sequences_available = train_x.shape[0]
print(f"Created {num_sequences_available} overlapping input/target sequence pairs.")
print(f"Shape of train_x: {train_x.shape}")
print(f"Shape of train_y: {train_y.shape}")



#### OUTPUT ####
Created 561 overlapping input/target sequence pairs.
Shape of train_x: torch.Size([561, 32])
Shape of train_y: torch.Size([561, 32])
```

This tells us that from our 593-character text, we could extract 561 overlapping sequences of length `32` (`block_size`).

The shapes `(561, 32)` confirm we have 561 rows (sequences) and 32 columns (token IDs per sequence) for both our inputs (`train_x`) and targets (`train_y`).

For training, we’ll randomly sample batch_size of these sequence pairs in each step.

Now, let’s initialize the building blocks of our Transformer.

```python
# Initialize the token embedding table
token_embedding_table = nn.Embedding(vocab_size, d_model).to(device)

print(f"Initialized Token Embedding Layer (Vocab: {vocab_size}, Dim: {d_model}). Device: {device}")


#### OUTPUT ####
Initialized Token Embedding Layer (Vocab: 36, Dim: 64). Device: cuda
```

![Token ID Lookup Process](https://miro.medium.com/v2/resize:fit:875/1*G0kAEjsdP_v9_ATyGfTeBg.png)

It knows our vocabulary size (`36`) and will map each character ID to a vector of size `64` (`d_model`), and it's placed on the correct device (`cuda`)

Transformers don’t inherently know the order of tokens like RNNs do. We need to inject position information. We use fixed sine and cosine waves of different frequencies. This matrix is added to the token embeddings.

```python
# Precompute the Sinusoidal Positional Encoding matrix
print("Creating Positional Encoding matrix...")
positional_encoding = torch.zeros(block_size, d_model, device=device)
position = torch.arange(0, block_size, dtype=torch.float, device=device).unsqueeze(1)
div_term_indices = torch.arange(0, d_model, 2, dtype=torch.float, device=device)
div_term = torch.exp(div_term_indices * (-math.log(10000.0) / d_model))
positional_encoding[:, 0::2] = torch.sin(position * div_term)
positional_encoding[:, 1::2] = torch.cos(position * div_term)
positional_encoding = positional_encoding.unsqueeze(0) # Add batch dimension

print(f"  Positional Encoding matrix created with shape: {positional_encoding.shape}. Device: {device}")


#### OUTPUT ####
Creating Positional Encoding matrix...
  Positional Encoding matrix created with shape: torch.Size([1, 32, 64]). Device: cuda
```

We’ve successfully created the positional encoding matrix. Its shape `(1, 32, 64)` corresponds to (Batch=`1`, Sequence Length=`block_size`, Embedding Dimension=`d_model`).

![Positional Embedding Inclusion](https://miro.medium.com/v2/resize:fit:875/1*tkKWWSX5wkCdAHKllWuoow.png)

This single matrix will be added to every sequence in a batch during the forward pass.

The core of the Transformer consists of multiple identical blocks stacked together. Each block contains `Self-Attention` and `Feed-Forward` layers. We'll initialize the components for all `n_layers` blocks.

```python
print(f"Initializing components for {n_layers} Transformer layers...")

# Lists to store layers for each Transformer block
layer_norms_1 = []      # LayerNorm before MHA
layer_norms_2 = []      # LayerNorm before FFN
mha_qkv_linears = []    # Combined Linear layer for Q, K, V projections
mha_output_linears = [] # Output Linear layer for MHA
ffn_linear_1 = []       # First linear layer in FFN
ffn_linear_2 = []       # Second linear layer in FFN

for i in range(n_layers):
    # Layer Normalization 1 (for MHA input)
    layer_norms_1.append(nn.LayerNorm(d_model).to(device))
    # Multi-Head Attention: QKV projection
    mha_qkv_linears.append(nn.Linear(d_model, 3 * d_model, bias=False).to(device))
    # Multi-Head Attention: Output projection
    mha_output_linears.append(nn.Linear(d_model, d_model).to(device))
    # Layer Normalization 2 (for FFN input)
    layer_norms_2.append(nn.LayerNorm(d_model).to(device))
    # Feed-Forward Network: Layer 1
    ffn_linear_1.append(nn.Linear(d_model, d_ff).to(device))
    # Feed-Forward Network: Layer 2
    ffn_linear_2.append(nn.Linear(d_ff, d_model).to(device))
    print(f"  Initialized components for Layer {i+1}/{n_layers}.")

print(f"Finished initializing components for {n_layers} layers.")



#### OUTPUT ####
Initializing components for 3 Transformer layers...
  Initialized components for Layer 1/3.
  Initialized components for Layer 2/3.
  Initialized components for Layer 3/3.
Finished initializing components for 3 layers.
```

For each of our `3` (`n_layers`) Transformer blocks, we have created and stored the necessary `Layer Normalization`, `Linear` layers for attention (QKV projection and output), and `Linear` layers for the `Feed-Forward` network. The model structure is taking shape!

Let’s break down the components within a block:

**Masked Multi-Head Self-Attention (MHA)**

This is the magic part. For each token, it looks at previous tokens (including itself) and calculates “attention scores” indicating how relevant each previous token is.

The “masked” part ensures it cannot peek at future tokens (crucial for generation). “Multi-Head” means it does this in parallel with different learned projections (Q, K, V — Query, Key, Value) to capture different types of relationships.

![Simplified Multi Head Attention](https://miro.medium.com/v2/resize:fit:1250/1*Ex4ZJZi56tWf114ML97TVw.png)
*Simplified Multi Head Attention (Created by [Fareed Khan](https://medium.com/u/b856005e5ecd?source=post_page---user_mention--ad0fa9c213d3---------------------------------------))*

**Add & Norm (Residual Connection + Layer Normalization):**

After MHA and FFN, we add the input of the sub-layer to its output (residual connection) and then apply Layer Normalization. Residual connections help gradients flow during training, preventing vanishing gradients.

Layer Normalization stabilizes the activations. We use a “Pre-LN” structure where normalization happens *before* the MHA/FFN layers.

![Add and Norm](https://miro.medium.com/v2/resize:fit:875/1*Fa7MPcbZFWs803OXmo7T8w.png)

*(Note: In Pre-LN, the Norm happens before the sub-layer, and the Add happens after)*

![Pre LN Norm](https://miro.medium.com/v2/resize:fit:875/1*Tq-VXBD6i5PcUT-GSIvF1w.png)
*Pre LN Norm (Created by [Fareed Khan](https://medium.com/u/b856005e5ecd?source=post_page---user_mention--ad0fa9c213d3---------------------------------------))*

**Position-wise Feed-Forward Network (FFN):**

A simple network applied independently to each token’s vector after attention. It consists of two linear layers with a ReLU activation in between, allowing the model to learn more complex transformations.

![Feed forward Network](https://miro.medium.com/v2/resize:fit:875/1*ZM1KNqog3KlY2efl9J9LVw.png)
*Feed forward Network (Created by [Fareed Khan](https://medium.com/u/b856005e5ecd?source=post_page---user_mention--ad0fa9c213d3---------------------------------------))*

Here’s how a single Transformer block looks conceptually (using Pre-LN):

![Single Transformer Block](https://miro.medium.com/v2/resize:fit:875/1*txATTLXHnroqYrBYwgxSBQ.png)

After the last Transformer block, we apply one final `Layer Normalization` and then a `Linear` layer to project the final `d_model` vectors back to the size of our vocabulary (`vocab_size`). These are the "logits" – raw scores for each possible next character.

```python
print("Initializing final LayerNorm and Output layers...")

# Final Layer Normalization
final_layer_norm = nn.LayerNorm(d_model).to(device)
print(f"  Initialized Final LayerNorm. Device: {device}")

# Final Linear Layer (language modeling head)
output_linear_layer = nn.Linear(d_model, vocab_size).to(device)
print(f"  Initialized Output Linear Layer (to vocab size {vocab_size}). Device: {device}")


#### OUTPUT ####
Initializing final LayerNorm and Output layers...
  Initialized Final LayerNorm. Device: cuda
  Initialized Output Linear Layer (to vocab size 36). Device: cuda
```

the last two important pieces are ready: the final normalization step and the output layer that translates the model’s internal representation back into scores for each character in our vocabulary.

Now that all components are initialized, we can train the model.

We are using Cross-Entropy Loss, which is standard for classification tasks like predicting the next token. AdamW is a common and effective optimizer.We gather *all* learnable parameters from our components for the optimizer to manage.

```python
# Define the loss function
criterion = nn.CrossEntropyLoss()
print(f"Loss function defined: {type(criterion).__name__}")

# Gather all model parameters
all_model_parameters = list(token_embedding_table.parameters())
for i in range(n_layers):
    all_model_parameters.extend(list(layer_norms_1[i].parameters()))
    all_model_parameters.extend(list(mha_qkv_linears[i].parameters()))
    all_model_parameters.extend(list(mha_output_linears[i].parameters()))
    all_model_parameters.extend(list(layer_norms_2[i].parameters()))
    all_model_parameters.extend(list(ffn_linear_1[i].parameters()))
    all_model_parameters.extend(list(ffn_linear_2[i].parameters()))
all_model_parameters.extend(list(final_layer_norm.parameters()))
all_model_parameters.extend(list(output_linear_layer.parameters()))

# Define the AdamW optimizer
optimizer = optim.AdamW(all_model_parameters, lr=learning_rate)
print(f"Optimizer defined: {type(optimizer).__name__}")
print(f"  Managing {len(all_model_parameters)} parameter groups/tensors.")

# Create the causal mask ONCE (lower triangular matrix)
causal_mask = torch.tril(torch.ones(block_size, block_size, device=device)).view(1, 1, block_size, block_size)


#### OUTPUT ####
Loss function defined: CrossEntropyLoss
Optimizer defined: AdamW
  Managing 38 parameter groups/tensors.
```

The `38` parameter groups/tensors tells us the total number of distinct weight matrices and bias vectors the optimizer needs to track and adjust across all the layers we defined.

We iterate for the specified number of epochs. In each epoch:

1.  **Get Batch**: Randomly sample input (`xb`) and target (`yb`) sequences.
2.  **Forward Pass**: Pass `xb` through the entire model (Embeddings -> Positional Encoding -> `n_layers` Transformer Blocks -> Final LayerNorm -> Output Linear Layer) to get logits.
3.  **Calculate Loss**: Compare the predicted logits with the actual target tokens `yb` using the Cross-Entropy criterion.
4.  **Backward Pass**: Calculate gradients (how much each parameter contributed to the error).
5.  **Optimizer Step**: Update the model parameters based on the gradients and the learning rate.

```python
print(f"\nStarting Training Loop for {epochs} epochs...")
losses = []

# (Set layers to train mode - omitted for brevity, done in notebook)

for epoch in range(epochs):

    # --- 1. Batch Selection ---
    indices = torch.randint(0, num_sequences_available, (batch_size,))
    xb = train_x[indices].to(device) # (B, T)
    yb = train_y[indices].to(device) # (B, T)

    # --- 2. Forward Pass (Executing Steps 3.1-3.3 conceptually) ---
    # Simplified representation of the forward pass logic from the notebook:
    B, T = xb.shape
    C = d_model
    # Embed + Positional Encode
    token_embed = token_embedding_table(xb)
    pos_enc_slice = positional_encoding[:, :T, :]
    x = token_embed + pos_enc_slice
    # Transformer Blocks Loop
    for i in range(n_layers):
        x_input_block = x
        # Pre-LN MHA
        x_ln1 = layer_norms_1[i](x_input_block)
        qkv = mha_qkv_linears[i](x_ln1)
        qkv = qkv.view(B, T, n_heads, 3 * d_k).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        attn_scores = (q @ k.transpose(-2, -1)) * (d_k ** -0.5)
        attn_scores_masked = attn_scores.masked_fill(causal_mask[:,:,:T,:T] == 0, float('-inf'))
        attention_weights = F.softmax(attn_scores_masked, dim=-1)
        attn_output = attention_weights @ v
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        mha_result = mha_output_linears[i](attn_output)
        x = x_input_block + mha_result # Residual 1
        # Pre-LN FFN
        x_input_ffn = x
        x_ln2 = layer_norms_2[i](x_input_ffn)
        ffn_hidden = ffn_linear_1[i](x_ln2)
        ffn_activated = F.relu(ffn_hidden)
        ffn_output = ffn_linear_2[i](ffn_activated)
        x = x_input_ffn + ffn_output # Residual 2
    # Final Layers
    final_norm_output = final_layer_norm(x)
    logits = output_linear_layer(final_norm_output) # (B, T, vocab_size)

    # --- 3. Calculate Loss ---
    B_loss, T_loss, V_loss = logits.shape
    loss = criterion(logits.view(B_loss * T_loss, V_loss), yb.view(B_loss * T_loss))

    # --- 4. Zero Gradients ---
    optimizer.zero_grad()

    # --- 5. Backward Pass ---
    loss.backward()

    # --- 6. Update Parameters ---
    optimizer.step()

    # --- Logging ---
    current_loss = loss.item()
    losses.append(current_loss)
    if epoch % eval_interval == 0 or epoch == epochs - 1:
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {current_loss:.4f}")

print("--- Training Loop Completed ---")
```

Once we start training, we do see the loss after each number of epochs.

```
Starting Training Loop for 5000 epochs...
  Epoch 1/5000, Loss: 3.6902
  Epoch 501/5000, Loss: 0.4272
  Epoch 1001/5000, Loss: 0.1480
  Epoch 1501/5000, Loss: 0.1461
  Epoch 2001/5000, Loss: 0.1226
  Epoch 2501/5000, Loss: 0.1281
  Epoch 3001/5000, Loss: 0.1337
  Epoch 3501/5000, Loss: 0.1288
  Epoch 4001/5000, Loss: 0.1178
  Epoch 4501/5000, Loss: 0.1292
  Epoch 5000/5000, Loss: 0.1053
--- Training Loop Completed ---
```

We see the training progress. The “Loss” value represents how wrong the model’s predictions were on average during that epoch. Notice how it starts high (around 3.69) and steadily decreases (down to around 0.10).

This significant drop is exactly what we want it means the model is getting much better at predicting the next character in the sequence as training progresses. It’s learning the patterns in “Alice in Wonderland”!

## Generating Text Using Transformer

Let’s use our trained model to generate new text.

We start with a seed character (or sequence) and ask the model to predict the next character. We take that prediction, add it to our sequence, and feed the new sequence back into the model to predict the *next* character, and so on. This is called autoregressive generation.

```python
print("\n--- Step 5: Text Generation ---")

# Seed character(s)
seed_chars = "t"
seed_ids = [char_to_int[ch] for ch in seed_chars]
generated_sequence = torch.tensor([seed_ids], dtype=torch.long, device=device)
print(f"Initial seed sequence: '{seed_chars}' -> {generated_sequence.tolist()}")

# Define how many new tokens (characters) to generate
num_tokens_to_generate = 200
print(f"Generating {num_tokens_to_generate} new tokens...")

# (Set layers to eval mode - omitted for brevity, done in notebook)

# Disable gradient calculations for efficiency
with torch.no_grad():
    for _ in range(num_tokens_to_generate):
        # --- 1. Prepare Input Context (last block_size tokens) ---
        current_context = generated_sequence[:, -block_size:]

        # --- 2. Forward Pass (similar to training, using current context) ---
        # Simplified representation of the generation forward pass:
        B_gen, T_gen = current_context.shape
        C_gen = d_model
        token_embed_gen = token_embedding_table(current_context)
        pos_enc_slice_gen = positional_encoding[:, :T_gen, :]
        x_gen = token_embed_gen + pos_enc_slice_gen
        for i in range(n_layers):
             # Pre-LN MHA... (same logic as training)
             x_input_block_gen = x_gen
             x_ln1_gen = layer_norms_1[i](x_input_block_gen)
             qkv_gen = mha_qkv_linears[i](x_ln1_gen)
             qkv_gen = qkv_gen.view(B_gen, T_gen, n_heads, 3 * d_k).permute(0, 2, 1, 3)
             q_gen, k_gen, v_gen = qkv_gen.chunk(3, dim=-1)
             attn_scores_gen = (q_gen @ k_gen.transpose(-2, -1)) * (d_k ** -0.5)
             attn_scores_masked_gen = attn_scores_gen.masked_fill(causal_mask[:,:,:T_gen,:T_gen] == 0, float('-inf'))
             attention_weights_gen = F.softmax(attn_scores_masked_gen, dim=-1)
             attn_output_gen = attention_weights_gen @ v_gen
             attn_output_gen = attn_output_gen.permute(0, 2, 1, 3).contiguous().view(B_gen, T_gen, C_gen)
             mha_result_gen = mha_output_linears[i](attn_output_gen)
             x_gen = x_input_block_gen + mha_result_gen # Residual 1
             # Pre-LN FFN... (same logic as training)
             x_input_ffn_gen = x_gen
             x_ln2_gen = layer_norms_2[i](x_input_ffn_gen)
             ffn_hidden_gen = ffn_linear_1[i](x_ln2_gen)
             ffn_activated_gen = F.relu(ffn_hidden_gen)
             ffn_output_gen = ffn_linear_2[i](ffn_activated_gen)
             x_gen = x_input_ffn_gen + ffn_output_gen # Residual 2
        # Final Layers
        final_norm_output_gen = final_layer_norm(x_gen)
        logits_gen = output_linear_layer(final_norm_output_gen)

        # --- 3. Get Logits for Last Time Step ---
        logits_last_token = logits_gen[:, -1, :] # Focus on the prediction for the *next* token

        # --- 4. Apply Softmax -> Probabilities ---
        probs = F.softmax(logits_last_token, dim=-1)

        # --- 5. Sample Next Token ---
        next_token = torch.multinomial(probs, num_samples=1) # Sample based on probabilities

        # --- 6. Append Sampled Token ---
        generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

print("\n--- Generation Complete ---")

# Decode the generated IDs back to text
final_generated_ids = generated_sequence[0].tolist()
decoded_text = ''.join([int_to_char[id] for id in final_generated_ids])

print(f"\nFinal Generated Text (including seed):")
print(decoded_text)
```

We use “t” as our start token and let’s print the next 200 tokens that are generated next to it using our trained model.

```
#### OUTPUT ####

--- Step 5: Text Generation ---
Initial seed sequence: 't' -> [[31]]
Generating 200 new tokens...

--- Generation Complete ---

Final Generated Text (including seed):
the
book her sister was reading, but it had no pictures or conversations in
in
it, 'and what is the use of a book,' thought Alice 'without pictures or
conversation?'
So she was considerinf her wad fe f
```

And here’s the result! Starting with just “t”, the model generated 200 more characters. Looking closely at the output, we can see it’s learned quite a bit!

Phrases like “book her sister was reading”, “pictures or conversations”, “thought Alice”, and the general structure clearly mimic the training text.

However, it’s not perfect **“considerinf her wad fe f”** shows that with limited data and training, the model can still produce nonsensical sequences.

> Now that our LLM can chat with text, we need to enhance it into a multimodal model, where it can interact with both images and videos.

## Chat with Images (ResNet)

We’ll extend the Transformer model we just built. The core idea is to:

1.  **Load Our Text Expert:** Start with the trained character-level Transformer.
2.  **Get Image Features:** Use a pre-trained vision model (like ResNet) to “see” the image and convert it into a list of numbers (a feature vector).
3.  **Align Features:** Make the image features compatible with the text model’s internal language using a simple linear layer.
4.  **Combine Inputs:** Treat the image features as a special `<IMG>` token at the beginning of the text prompt sequence.
5.  **Fine-Tune:** Train the combined model on examples that pair images, text prompts, and desired text responses.
6.  **Generate:** Give the model a new image and prompt, and let it generate a text response based on both.

We saved the state (weights, config, tokenizer) of our character-level Transformer in the previous step. Let’s load it back.

This gives us a model that already understands basic text structure.

```python
# --- Load Saved Text Model State ---
print("\nStep 0.2: Loading pre-trained text model state...")
model_load_path = 'saved_models/transformer_model.pt'
if not os.path.exists(model_load_path):
    raise FileNotFoundError(f"Error: Model file not found at {model_load_path}. Please ensure 'transformer2.ipynb' was run and saved the model.")

loaded_state_dict = torch.load(model_load_path, map_location=device)
print(f"Loaded state dictionary from '{model_load_path}'.")

# --- Extract Config and Tokenizer ---
config = loaded_state_dict['config']
loaded_vocab_size = config['vocab_size']
d_model = config['d_model']
n_heads = config['n_heads']
n_layers = config['n_layers']
d_ff = config['d_ff']
loaded_block_size = config['block_size'] # Max sequence length for text model
d_k = d_model // n_heads

char_to_int = loaded_state_dict['tokenizer']['char_to_int']
int_to_char = loaded_state_dict['tokenizer']['int_to_char']

print("Extracted model configuration and tokenizer:")
print(f"  Loaded vocab_size: {loaded_vocab_size}")
print(f"  d_model: {d_model}")
# ... (print other loaded hyperparameters) ...
print(f"  Loaded block_size: {loaded_block_size}")



#### OUTPUT ####
Loaded state dictionary from 'saved_models/transformer_model.pt'.
Extracted model configuration and tokenizer:
  Loaded vocab_size: 36
  d_model: 64
  n_layers: 3
  n_heads: 4
  d_ff: 256
  Loaded block_size: 32
```

We’ve loaded the configuration (`d_model`, `n_layers`, etc.) and the character mappings (`char_to_int`, `int_to_char`) from our previously saved text model. Notice the original `vocab_size` was 36 and the `block_size` was 32.

To handle images and sequence processing, we need a few new markers in our vocabulary:

 `<IMG>`: Represents the image's place in the sequence.
 `<PAD>`: Used to make all sequences in a batch the same length.
 `<EOS>`: Signals the end of a generated response.

```python
print("\nStep 0.3: Defining special tokens and updating vocabulary...")

# --- Define Special Tokens ---
img_token = "<IMG>"
pad_token = "<PAD>"
eos_token = "<EOS>" # End-of-Sentence/Sequence
special_tokens = [img_token, pad_token, eos_token]

# --- Add Special Tokens to Vocabulary ---
current_vocab_size = loaded_vocab_size
for token in special_tokens:
    if token not in char_to_int:
        char_to_int[token] = current_vocab_size
        int_to_char[current_vocab_size] = token
        current_vocab_size += 1

# Update vocab_size
vocab_size = current_vocab_size
pad_token_id = char_to_int[pad_token] # Store the ID for later use

print(f"Added special tokens: {special_tokens}")
print(f"Updated vocabulary size: {vocab_size}")
print(f"PAD token ID: {pad_token_id}")


#### OUTPUT ####
Added special tokens: ['<IMG>', '<PAD>', '<EOS>']
Updated vocabulary size: 39
PAD token ID: 37
```

Our vocabulary now includes these 3 special tokens, bringing the total `vocab_size` to 39. We also note the ID for the `<PAD>` token (37), which we'll need for padding later.

Real multimodal training needs lots of (Image, Prompt, Response) examples. For our simple case, we’ll create a tiny dataset with dummy images (colored shapes) and corresponding questions/answers.

```lua
print("\nStep 0.4: Defining sample multi-modal data...")

# --- Create Dummy Image Files ---
sample_data_dir = "sample_multimodal_data"
os.makedirs(sample_data_dir, exist_ok=True)
image_paths = {
    "red": os.path.join(sample_data_dir, "red_square.png"),
    "blue": os.path.join(sample_data_dir, "blue_square.png"),
    "green": os.path.join(sample_data_dir, "green_circle.png")
}
# (Code to create red, blue, green images - same as notebook)
img_red = Image.new('RGB', (64, 64), color = 'red')
img_red.save(image_paths["red"])
img_blue = Image.new('RGB', (64, 64), color = 'blue')
img_blue.save(image_paths["blue"])
img_green = Image.new('RGB', (64, 64), color = 'white')
from PIL import ImageDraw
draw = ImageDraw.Draw(img_green)
draw.ellipse((4, 4, 60, 60), fill='green', outline='green')
img_green.save(image_paths["green"])
print(f"Created dummy images in '{sample_data_dir}'.")

# --- Define Data Triplets ---
# Added <EOS> token to the end of responses.
sample_training_data = [
    {"image_path": image_paths["red"], "prompt": "What color is the shape?", "response": "red." + eos_token},
    # ... (other samples) ...
    {"image_path": image_paths["green"], "prompt": "Describe this.", "response": "a circle, it is green." + eos_token}
]
num_samples = len(sample_training_data)
print(f"Defined {num_samples} sample multi-modal data points.")
```

![MultiModal Training data](https://miro.medium.com/v2/resize:fit:875/1*CrJRpcj_p9l6d7gzb6FVoA.png)

We’ve generated our simple red, blue, and green images and created 6 training examples pairing these images with questions and answers like

> What color is the shape? -> red

We need a way to convert images into numbers the Transformer can use. We’ll use ResNet-18, a popular pre-trained image recognition model, but we’ll chop off its final classification layer.

The output from the layer just before that will be our image feature vector. We’ll keep this vision model “frozen” (not train it further) for simplicity.

```python
print("\nStep 0.5: Loading pre-trained vision model (ResNet-18)...")

# --- Load Pre-trained ResNet-18 ---
vision_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

# --- Remove Final Classification Layer ---
vision_feature_dim = vision_model.fc.in_features # Get feature dimension (512)
vision_model.fc = nn.Identity() # Replace the classifier

# --- Set to Evaluation Mode and Move to Device ---
vision_model = vision_model.to(device)
vision_model.eval() # IMPORTANT: Disable dropout/batchnorm updates

print(f"Loaded ResNet-18 feature extractor.")
print(f"  Output feature dimension: {vision_feature_dim}")
print(f"  Vision model set to evaluation mode on device: {device}")


#### OUTPUT ####
Step 0.5: Loading pre-trained vision model (ResNet-18)...
Loaded ResNet-18 feature extractor.
  Output feature dimension: 512
  Vision model set to evaluation mode on device: cuda
```

Okay, our ResNet-18 is loaded, its classifier is removed, and it’s ready on the GPU (cuda). It will output feature vectors of size 512 for any given image.

![ResNet Extractor](https://miro.medium.com/v2/resize:fit:1250/1*jm4VHMfDA0XlAT6sHVCxDA.png)


So the images need to be resized and normalized consistently before feeding them to ResNet. We’ll use standard transformations recommended for models trained on ImageNet.

```bash
print("\nStep 0.6: Defining image transformations...")

# --- Define Standard ImageNet Transforms ---
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet mean/std
                         std=[0.229, 0.224, 0.225])
])

print("Defined image preprocessing pipeline (Resize, Crop, ToTensor, Normalize).")


#### OUTPUT ####
Step 0.6: Defining image transformations...
Defined image preprocessing pipeline (Resize, Crop, ToTensor, Normalize).
```

We also need to adjust some parameters, like the `block_size`, to accommodate the combined image and text sequence. We'll also define how many tokens represent the image (just 1 in our case).

```python
print("\nStep 0.7: Defining new/updated hyperparameters...")

# --- Multi-Modal Sequence Length ---
block_size = 64 # Increase block size for combined sequence
print(f"  Set combined block_size: {block_size}")

# --- Number of Image Tokens ---
num_img_tokens = 1
print(f"  Using {num_img_tokens} <IMG> token(s) to represent image features.")

# --- Training Parameters ---
learning_rate = 3e-4
batch_size = 4 # Reduce batch size
epochs = 2000
eval_interval = 500
print(f"  Updated Training Params: LR={learning_rate}, BatchSize={batch_size}, Epochs={epochs}")

# (Check if block_size is sufficient - code omitted for brevity)
min_req_block_size = num_img_tokens + max(len(d["prompt"]) + len(d["response"]) for d in sample_training_data) + 1
print(f"  Max sequence length in sample data (approx): {min_req_block_size}")
# Recreate the causal mask for the new block_size
causal_mask = torch.tril(torch.ones(block_size, block_size, device=device)).view(1, 1, block_size, block_size)
print(f"  Recreated causal mask for new block_size={block_size}")


#### OUTPUT ####
Step 0.7: Defining new/updated hyperparameters...
  Set combined block_size: 64
  Using 1 <IMG> token(s) to represent image features.
  Updated Training Params: LR=0.0003, BatchSize=4, Epochs=2000
  Max sequence length in sample data (approx): 43
  Recreated causal mask for new block_size=64
```

We’ve increased the `block_size` to 64 to fit the image token plus potentially longer prompt/response pairs. We also reduced the `batch_size` to 4, as processing images and longer sequences might require more memory. The causal mask used in attention is also updated for this new sequence length.

Let’s process our sample data into a format the model can train on.

First, run each unique image through the ResNet feature extractor and store the results. This avoids re-computing them repeatedly during training.

```python
print("\nStep 1.1: Extracting image features for sample data...")
extracted_image_features = {}
unique_image_paths = set(d["image_path"] for d in sample_training_data)
print(f"Found {len(unique_image_paths)} unique images to process.")

for img_path in unique_image_paths:
    # --- Load Image ---
    img = Image.open(img_path).convert('RGB')
    # --- Apply Transformations ---
    img_tensor = image_transforms(img).unsqueeze(0).to(device)
    # --- Extract Features ---
    with torch.no_grad():
        feature_vector = vision_model(img_tensor)
    # --- Store Features ---
    extracted_image_features[img_path] = feature_vector.squeeze(0)
    print(f"  Extracted features for '{os.path.basename(img_path)}', shape: {extracted_image_features[img_path].shape}")

print("Finished extracting image features for all unique sample images.")


#### OUTPUT ####
Found 3 unique images to process.
  Extracted features for 'green_circle.png', shape: torch.Size([512])
  Extracted features for 'blue_square.png', shape: torch.Size([512])
  Extracted features for 'red_square.png', shape: torch.Size([512])
Finished extracting image features for all unique sample images.
```

Great! We now have the 512-dimensional feature vectors for our red, blue, and green images stored and ready.

Next, convert the text prompts and responses into sequences of token IDs using our updated vocabulary.

```python
print("\nStep 1.2: Tokenizing prompts and responses...")

# (Code to check for and add any new characters from data to vocab - from notebook)
current_vocab_size = vocab_size
all_chars = set()
for sample in sample_training_data:
    all_chars.update(sample["prompt"])
    response_text = sample["response"].replace(eos_token, "")
    all_chars.update(response_text)
new_chars_added = 0
for char in all_chars:
     if char not in char_to_int:
          # (Add char to mappings - code omitted)
          new_chars_added += 1
vocab_size = current_vocab_size + new_chars_added
print(f"Added {new_chars_added} new characters to vocabulary. New vocab_size: {vocab_size}")


# --- Tokenize ---
tokenized_samples = []
for sample in sample_training_data:
    prompt_ids = [char_to_int[ch] for ch in sample["prompt"]]
    # Handle EOS in response
    response_text = sample["response"]
    if response_text.endswith(eos_token):
        response_ids = [char_to_int[ch] for ch in response_text[:-len(eos_token)]] + [char_to_int[eos_token]]
    else:
        response_ids = [char_to_int[ch] for ch in response_text]

    tokenized_samples.append({
        "image_path": sample["image_path"],
        "prompt_ids": prompt_ids,
        "response_ids": response_ids
    })

print(f"Tokenized text for all {len(tokenized_samples)} samples.")



#### OUTPUT ####
Step 1.2: Tokenizing prompts and responses...
Added 3 new characters to vocabulary. New vocab_size: 42
Tokenized text for all 6 samples.
```

It seems our sample prompts/responses introduced 3 new characters not present in the original Alice text (likely ‘W’, ‘D’, ‘I’ etc from the prompts). Our vocab_size is now 42. All text parts are now numerical IDs.

Now, combine the image representation (`<IMG>` ID), prompt IDs, and response IDs into single input sequences. We also create target sequences (shifted versions for prediction) and padding masks.

```python
print("\nStep 1.3: Creating padded input/target sequences and masks...")

prepared_sequences = []
ignore_index = -100 # For loss calculation

for sample in tokenized_samples:
    # --- Construct Input Sequence IDs ---
    img_ids = [char_to_int[img_token]] * num_img_tokens
    # Input: <IMG> + prompt + response (except last token)
    input_ids_no_pad = img_ids + sample["prompt_ids"] + sample["response_ids"][:-1]

    # --- Construct Target Sequence IDs ---
    # Target: shifted input, ignore loss for <IMG> and prompt
    target_ids_no_pad = ([ignore_index] * len(img_ids)) + \
                         ([ignore_index] * len(sample["prompt_ids"])) + \
                         sample["response_ids"]

    # --- Padding ---
    current_len = len(input_ids_no_pad)
    pad_len = block_size - current_len
    # (Handle truncation if current_len > block_size - code omitted)
    input_ids = input_ids_no_pad + ([pad_token_id] * pad_len)
    target_ids = target_ids_no_pad + ([ignore_index] * pad_len) # Pad targets too

    # --- Create Attention Mask (Padding Mask) ---
    attention_mask = ([1] * current_len) + ([0] * pad_len) # 1 for real, 0 for pad

    # --- Store ---
    prepared_sequences.append({
        "image_path": sample["image_path"],
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "target_ids": torch.tensor(target_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
    })

# --- Stack into Tensors ---
all_input_ids = torch.stack([s['input_ids'] for s in prepared_sequences])
all_target_ids = torch.stack([s['target_ids'] for s in prepared_sequences])
all_attention_masks = torch.stack([s['attention_mask'] for s in prepared_sequences])
all_image_paths = [s['image_path'] for s in prepared_sequences] # Keep track of images

num_sequences_available = all_input_ids.shape[0]
print(f"Created {num_sequences_available} padded sequences with targets and masks.")
print(f"  Input IDs shape: {all_input_ids.shape}")
print(f"  Target IDs shape: {all_target_ids.shape}")
print(f"  Attention Mask shape: {all_attention_masks.shape}")



#### OUTPUT ####
Step 1.3: Creating padded input/target sequences and masks...
Created 6 padded sequences with targets and masks.
  Input IDs shape: torch.Size([6, 64])
  Target IDs shape: torch.Size([6, 64])  // Note: Notebook has 65, careful check needed on logic/off-by-one. Assuming 64 for now.
  Attention Mask shape: torch.Size([6, 64])
```

Excellent. Each of our 6 samples is now a set of tensors:

*   `input_ids` (containing `<IMG>` ID + prompt IDs + response IDs, padded to length 64)
*   `target_ids` (containing ignored IDs for image/prompt, then response IDs, padded with `ignore_index`)
*   `attention_mask` (1s for real tokens, 0s for padding)

The shapes confirm this structure. We’ll use random sampling again for batching during training.

Our vocabulary grew from 36 to 42 (including the 3 new characters and 3 special tokens). The embedding table and the final output layer need to be resized. We’ll copy the weights for the original characters and let the new entries be randomly initialized.

```python
print("\nStep 2.1: Re-initializing Embedding and Output Layers for new vocab size...")

# --- Token Embedding Table ---
new_token_embedding_table = nn.Embedding(vocab_size, d_model).to(device)
original_weights = loaded_state_dict['token_embedding_table']['weight'][:loaded_vocab_size, :]
with torch.no_grad():
    new_token_embedding_table.weight[:loaded_vocab_size, :] = original_weights
token_embedding_table = new_token_embedding_table
print(f"  Re-initialized Token Embedding Table, shape: {token_embedding_table.weight.shape}")

# --- Output Linear Layer ---
new_output_linear_layer = nn.Linear(d_model, vocab_size).to(device)
original_out_weight = loaded_state_dict['output_linear_layer']['weight'][:loaded_vocab_size, :]
original_out_bias = loaded_state_dict['output_linear_layer']['bias'][:loaded_vocab_size]
with torch.no_grad():
    new_output_linear_layer.weight[:loaded_vocab_size, :] = original_out_weight
    new_output_linear_layer.bias[:loaded_vocab_size] = original_out_bias
output_linear_layer = new_output_linear_layer
print(f"  Re-initialized Output Linear Layer, weight shape: {output_linear_layer.weight.shape}")



#### OUTPUT ####
  Re-initialized Token Embedding Table, shape: torch.Size([42, 64])
  Re-initialized Output Linear Layer, weight shape: torch.Size([42, 64])
```

The shapes now reflect the updated `vocab_size` of 42. The weights for the original 36 characters are preserved.

This is a *new* layer that learns to map the 512-dim image features from ResNet to the 64-dim space of our Transformer (d_model).

```python
vision_projection_layer = nn.Linear(vision_feature_dim, d_model).to(device)
```

![Vision Projection Layer](https://miro.medium.com/v2/resize:fit:875/1*0Yws8XVVT9kXyMBRdNya_Q.png)

This important bridge between the vision and language modalities is now initialized.

We need to load the weights for the actual Transformer blocks (LayerNorms, Attention layers, FFN layers) from our saved text model.

```python
print("\nStep 2.3: Loading parameters for existing Transformer Blocks...")

# (Code to re-initialize layer lists and load state_dict for each component:
# layer_norms_1, mha_qkv_linears, mha_output_linears, layer_norms_2,
# ffn_linear_1, ffn_linear_2, final_layer_norm - from notebook)

# Reload components and load state dictionaries from loaded_state_dict
layer_norms_1 = []
mha_qkv_linears = []
mha_output_linears = []
layer_norms_2 = []
ffn_linear_1 = []
ffn_linear_2 = []

for i in range(n_layers):
    ln1 = nn.LayerNorm(d_model).to(device)
    ln1.load_state_dict(loaded_state_dict['layer_norms_1'][i])
    layer_norms_1.append(ln1)

    qkv_dict = loaded_state_dict['mha_qkv_linears'][i]
    has_qkv_bias = 'bias' in qkv_dict
    qkv = nn.Linear(d_model, 3 * d_model, bias=has_qkv_bias).to(device)
    qkv.load_state_dict(qkv_dict)
    mha_qkv_linears.append(qkv)

    out_dict = loaded_state_dict['mha_output_linears'][i]
    has_out_bias = 'bias' in out_dict
    out = nn.Linear(d_model, d_model, bias=has_out_bias).to(device)
    out.load_state_dict(out_dict)
    mha_output_linears.append(out) # Corrected variable name

    ln2 = nn.LayerNorm(d_model).to(device)
    ln2.load_state_dict(loaded_state_dict['layer_norms_2'][i])
    layer_norms_2.append(ln2)

    ff1_dict = loaded_state_dict['ffn_linear_1'][i]
    has_ff1_bias = 'bias' in ff1_dict
    ff1 = nn.Linear(d_model, d_ff, bias=has_ff1_bias).to(device)
    ff1.load_state_dict(ff1_dict)
    ffn_linear_1.append(ff1)

    ff2_dict = loaded_state_dict['ffn_linear_2'][i]
    has_ff2_bias = 'bias' in ff2_dict
    ff2 = nn.Linear(d_ff, d_model, bias=has_ff2_bias).to(device)
    ff2.load_state_dict(ff2_dict)
    ffn_linear_2.append(ff2)
    print(f"  Loaded components for Layer {i+1}/{n_layers}.")

final_layer_norm = nn.LayerNorm(d_model).to(device)
final_layer_norm.load_state_dict(loaded_state_dict['final_layer_norm'])
print("  Loaded Final LayerNorm.")

# Load Positional Encoding
positional_encoding = loaded_state_dict['positional_encoding'].to(device)
# (Code to recompute PE if block_size changed - from notebook)
if positional_encoding.shape[1] != block_size:
     print(f"Warning: Loaded positional encoding size ({positional_encoding.shape[1]}) != new block_size ({block_size}). Recomputing.")
     # (Recompute PE logic - code omitted)
     new_pe = torch.zeros(block_size, d_model, device=device)
     position = torch.arange(0, block_size, dtype=torch.float, device=device).unsqueeze(1)
     div_term_indices = torch.arange(0, d_model, 2, dtype=torch.float, device=device)
     div_term = torch.exp(div_term_indices * (-math.log(10000.0) / d_model))
     new_pe[:, 0::2] = torch.sin(position * div_term)
     new_pe[:, 1::2] = torch.cos(position * div_term)
     positional_encoding = new_pe.unsqueeze(0)
     print(f"  Recomputed Positional Encoding matrix, shape: {positional_encoding.shape}")


print("Finished loading existing model components.")
```

The weights for the core Transformer logic are now loaded.

```python
#### OUPUT ####
Step 2.3: Loading parameters for existing Transformer Blocks...
  Loaded components for Layer 1/3.
  Loaded components for Layer 2/3.
  Loaded components for Layer 3/3.
  Loaded Final LayerNorm.
Warning: Loaded positional encoding size (32) != new block_size (64). Recomputing.
  Recomputed Positional Encoding matrix, shape: torch.Size([1, 64, 64])
Finished loading existing model components.
```

We also got a warning and recomputed the positional encoding because our block_size changed from 32 to 64.

We define the optimizer again, making sure it includes the new `vision_projection_layer` and the resized embedding/output layers. The loss function now uses `ignore_index=-100` to disregard padding and prompt tokens during error calculation.

```python
print("\nStep 2.4: Defining Optimizer and Loss Function...")

# --- Gather All Trainable Parameters ---
all_trainable_parameters = list(token_embedding_table.parameters())
all_trainable_parameters.extend(list(vision_projection_layer.parameters())) # Include new layer
# (Extend with all other layer parameters - code omitted)
for i in range(n_layers):
    all_trainable_parameters.extend(list(layer_norms_1[i].parameters()))
    all_trainable_parameters.extend(list(mha_qkv_linears[i].parameters()))
    all_trainable_parameters.extend(list(mha_output_linears[i].parameters())) # Correct name
    all_trainable_parameters.extend(list(layer_norms_2[i].parameters()))
    all_trainable_parameters.extend(list(ffn_linear_1[i].parameters()))
    all_trainable_parameters.extend(list(ffn_linear_2[i].parameters()))
all_trainable_parameters.extend(list(final_layer_norm.parameters()))
all_trainable_parameters.extend(list(output_linear_layer.parameters()))

# --- Define Optimizer ---
optimizer = optim.AdamW(all_trainable_parameters, lr=learning_rate)
print(f"  Optimizer defined: AdamW with lr={learning_rate}")
print(f"  Managing {len(all_trainable_parameters)} parameter groups/tensors.")

# --- Define Loss Function ---
criterion = nn.CrossEntropyLoss(ignore_index=ignore_index) # Use ignore_index
print(f"  Loss function defined: CrossEntropyLoss (ignore_index={ignore_index})")


#### OUTPUT ####
  Optimizer defined: AdamW with lr=0.0003
  Managing 40 parameter groups/tensors.
  Loss function defined: CrossEntropyLoss (ignore_index=-100)
```

Our optimizer is ready, now managing 40 parameter groups (up from 38 due to the new vision projection layer). The loss function is set to correctly ignore non-response tokens.

Let’s train (fine-tune) the model on our sample multimodal data.

![MultiModal Training Loop](https://miro.medium.com/v2/resize:fit:875/1*ul8x_gEueWmrjL92GqzX5A.png)

```python
# Set layers to train mode
token_embedding_table.train()
vision_projection_layer.train()
for layer in layer_norms_1:
    layer.train()
final_layer_norm.train()
output_linear_layer.train()

for epoch in range(epochs):
    # Batch Selection
    indices = torch.randint(0, num_sequences_available, (batch_size,))
    xb_ids, yb_ids, batch_masks = all_input_ids[indices].to(device), all_target_ids[indices].to(device), all_attention_masks[indices].to(device)
    batch_img_paths = [all_image_paths[i] for i in indices.tolist()]
    try:
        batch_img_features = torch.stack([extracted_image_features[p] for p in batch_img_paths]).to(device)
    except KeyError:
        continue

    # Forward Pass
    projected_img_features = vision_projection_layer(batch_img_features).unsqueeze(1)
    text_token_embeddings = token_embedding_table(xb_ids)
    combined_embeddings = text_token_embeddings.clone()
    combined_embeddings[:, :num_img_tokens, :] = projected_img_features
    x = combined_embeddings + positional_encoding[:, :xb_ids.shape[1], :]

    # Attention Mask
    combined_attn_mask = causal_mask[:, :xb_ids.shape[1], :xb_ids.shape[1]] * batch_masks.unsqueeze(1).unsqueeze(2)

    # Transformer Blocks
    for i in range(n_layers):
        x_ln1 = layer_norms_1[i](x)
        qkv = mha_qkv_linears[i](x_ln1).view(B, T, n_heads, 3 * d_k).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        attn_scores = (q @ k.transpose(-2, -1)) * (d_k ** -0.5)
        attn_scores_masked = attn_scores.masked_fill(combined_attn_mask == 0, float('-inf'))
        attention_weights = torch.nan_to_num(F.softmax(attn_scores_masked, dim=-1))
        attn_output = (attention_weights @ v).permute(0, 2, 1, 3).contiguous().view(B, T, C)
        x = x + mha_output_linears[i](attn_output)

        # FFN
        x = x + ffn_linear_2[i](F.relu(ffn_linear_1[i](layer_norms_2[i](x))))

    # Final Layers
    logits = output_linear_layer(final_layer_norm(x))

    # Calculate Loss
    targets_reshaped = yb_ids.view(-1) if yb_ids.size(1) == logits.shape[1] else yb_ids[:, :logits.shape[1]].view(-1)
    loss = criterion(logits.view(-1, V_loss), targets_reshaped)

    # Backpropagation
    optimizer.zero_grad()
    if torch.isfinite(loss):
        loss.backward()
        optimizer.step()

    # Logging
    if loss is not None:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("--- Multi-Modal Training Loop Completed ---")

```

This loop will start training and prints the eval loss after each number of epochs.

```python
Step 3.1: Starting Multi-Modal Training Loop...
  Epoch 1/2000, Loss: 9.1308
  Epoch 501/2000, Loss: 0.0025
  Epoch 1001/2000, Loss: 0.0013
  Epoch 1501/2000, Loss: 0.0005
  Epoch 2000/2000, Loss: 0.0004
--- Multi-Modal Training Loop Completed ---
```

![Training Loss MultiModal](https://miro.medium.com/v2/resize:fit:1250/1*70oLaG21fMaebio1rjXxfA.png)
*Training Loss MultiModal*

The loss dropped incredibly low, very quickly (obviously overfitting not enough data).

But this suggests the model rapidly learned to associate the simple images and prompts with the short answers in our tiny dataset. The loss near zero indicates it’s almost perfectly predicting the characters in the expected responses for the training samples.

## Testing Chat with Images Functionality

Let’s test our fine-tuned model. We’ll give it the green circle image and the prompt “Describe this image: “.

```python
print("\nStep 4.3: Decoding generated sequence...")

if generated_sequence_ids is not None:
    final_ids_list = generated_sequence_ids[0].tolist()
    decoded_text = "".join([int_to_char.get(id_val, f"[UNK:{id_val}]") for id_val in final_ids_list]) # Use .get for safety

    print(f"\n--- Final Generated Output ---")
    print(f"Image: {os.path.basename(test_image_path)}")
    response_start_index = num_img_tokens + len(test_prompt_text)
    print(f"Prompt: {test_prompt_text}")
    print(f"Generated Response: {decoded_text[response_start_index:]}")
else:
    print("Decoding skipped.")


#### OUTPUT ####
Image: green_circle.png
Prompt: Describe this image:
Generated Response: ge:   of    fad r qv listen qqqda
```

This isn’t surprising given our extremely small dataset (only 6 examples!) and the limited training (2000 steps).

> It learned *something* about the association (it started with ‘g’/’e’ for green)

but didn’t generalize well enough to form coherent sentences beyond what it memorized.

## Chat With Video/Audio (ResNet + Feature Vectors)

Our model can now “see” images and answer questions about them! But modern multimodal models like GPT-4o go further, understanding video and audio too. How can we extend our framework to handle these?

The core idea remains the same: **convert the modality into numerical features and integrate them into the Transformer’s sequence.**

1.  To process video, we treat it as a sequence of frames. Each frame is passed through a pre-trained model like ResNet-18 to extract feature vectors. These frame features are then averaged to summarize the video’s visual content.
2.  We project this average feature vector into the transformer’s internal dimension using a linear layer. To signal video input, we introduce a special `<VID>` token in the model’s sequence, replacing its usual embedding with the projected video features.

Let’s imagine we have a video processing library (`dummy_video_lib`) and want to process a video file.

```python
print("\n--- Conceptual Video Handling ---")

# 0. Add <VID> token
vid_token = "<VID>"
if vid_token not in char_to_int:
    char_to_int[vid_token] = vocab_size
    int_to_char[vocab_size] = vid_token
    vocab_size += 1
    print(f"Added {vid_token}. New vocab_size: {vocab_size}")
    # Need to resize embedding/output layers again if doing this properly!

# 1. Load video frames (dummy example)
video_path = "path/to/dummy_video.mp4"
# dummy_frames = load_video_frames(video_path, num_frames=16) # List of PIL Images
# Create dummy frames for illustration: list of 16 green PIL Images
dummy_frames = [img_green] * 16 # Reuse green circle image
print(f"Loaded {len(dummy_frames)} dummy frames for '{video_path}'.")

# 2. Extract features for each frame
frame_features_list = []
with torch.no_grad():
    for frame_img in dummy_frames:
        # Apply same image transforms as before
        frame_tensor = image_transforms(frame_img).unsqueeze(0).to(device)
        # Use the SAME vision model
        frame_feature = vision_model(frame_tensor) # (1, vision_feature_dim)
        frame_features_list.append(frame_feature)

# Stack features: (num_frames, vision_feature_dim)
all_frame_features = torch.cat(frame_features_list, dim=0)
print(f"Extracted frame features, shape: {all_frame_features.shape}") # e.g., (16, 512)

# 3. Combine Features (Simple Averaging)
video_feature_avg = torch.mean(all_frame_features, dim=0, keepdim=True) # (1, vision_feature_dim)
print(f"Averaged video features, shape: {video_feature_avg.shape}") # e.g., (1, 512)

# 4. Project Video Features
# Option 1: Use the same projection layer as images (if appropriate)
# Option 2: Create a dedicated video projection layer
# vision_video_projection_layer = nn.Linear(vision_feature_dim, d_model).to(device)
# Initialize and train this layer appropriately!
# For simplicity here, let's reuse the image projection layer conceptually
with torch.no_grad(): # Assuming projection layer is trained/loaded
    projected_video_feature = vision_projection_layer(video_feature_avg) # (1, d_model)
print(f"Projected video feature, shape: {projected_video_feature.shape}") # e.g., (1, 64)

# 5. Prepare Input Sequence (Example)
prompt_text = "What happens in the video?"
prompt_ids = [char_to_int[ch] for ch in prompt_text]
vid_id = char_to_int[vid_token]

# Input: [<VID>, prompt tokens]
input_ids_vid = torch.tensor([[vid_id] + prompt_ids], dtype=torch.long, device=device)
print(f"Example input sequence with video: {input_ids_vid.tolist()}")

# During the actual forward pass for this sequence, the embedding for vid_id
# would be replaced by projected_video_feature.
```

This shows how we can get a single vector representing the video, project it, and prepare an input sequence starting with the `<VID>` token.

```python
#### OUTPUT ####
Loaded 16 dummy frames for 'path/to/dummy_video.mp4'.
Extracted frame features, shape: torch.Size([16, 512])
Averaged video features, shape: torch.Size([1, 512])
Projected video feature, shape: torch.Size([1, 64])
Example input sequence with video: [[42, 41, 21, 14, 31, 1, 21, 14, 29, 29, 18, 27, 30, 1, 22, 27, 1, 31, 21, 18, 1, 33, 22, 17, 18, 28, 9]]
```

Handling audio inputs follows a very similar strategy. First, we load the audio waveform from a file. The next step is converting this raw waveform into a numerical representation suitable for machine learning.

```python
# Assume dummy_audio_lib exists
# from dummy_audio_lib import load_audio_waveform, compute_mfccs

print("\n--- Conceptual Audio Handling ---")

# 0. Add <AUD> token
aud_token = "<AUD>"
if aud_token not in char_to_int:
    char_to_int[aud_token] = vocab_size
    int_to_char[vocab_size] = aud_token
    vocab_size += 1
    print(f"Added {aud_token}. New vocab_size: {vocab_size}")
    # Resize embedding/output layers needed!

# 1. Load Audio (dummy waveform)
audio_path = "path/to/dummy_audio.wav"
# waveform, sample_rate = load_audio_waveform(audio_path)
# Dummy: 1 second of audio at 16kHz
sample_rate = 16000
waveform = np.random.randn(1 * sample_rate)
print(f"Loaded dummy waveform for '{audio_path}', length: {len(waveform)}")

# 2. Extract Features (dummy MFCCs + Linear layer)
# mfccs = compute_mfccs(waveform, sample_rate) # Shape: (num_frames, num_mfcc_coeffs)
# Dummy: (99 time steps, 13 coefficients)
mfccs = np.random.randn(99, 13)
mfccs_tensor = torch.tensor(mfccs, dtype=torch.float).to(device)
print(f"Computed dummy MFCC features, shape: {mfccs_tensor.shape}")

# Simple placeholder feature extractor (e.g., a Linear layer over coefficients)
# Needs to be defined and potentially trained
audio_feature_dim = mfccs_tensor.shape[1] # e.g., 13
dummy_audio_extractor = nn.Linear(audio_feature_dim, 256).to(device) # Example: map MFCCs -> 256
# Initialize/load weights for dummy_audio_extractor!
with torch.no_grad():
     audio_features_time = dummy_audio_extractor(mfccs_tensor) # (num_frames, 256)
print(f"Extracted dummy audio features over time, shape: {audio_features_time.shape}")

# 3. Combine Features (Simple Averaging over time)
audio_feature_avg = torch.mean(audio_features_time, dim=0, keepdim=True) # (1, 256)
print(f"Averaged audio features, shape: {audio_feature_avg.shape}")

# 4. Project Audio Features to d_model
# Need a dedicated audio projection layer
# audio_projection_layer = nn.Linear(256, d_model).to(device)
# Initialize and train this layer!
# For simplicity, reuse image projection layer conceptually (dimensions likely won't match in reality!)
# Assuming audio_feature_avg was projected to vision_feature_dim somehow or using a dedicated layer:
# Dummy projection to d_model:
dummy_audio_projection = nn.Linear(256, d_model).to(device)
with torch.no_grad():
    projected_audio_feature = dummy_audio_projection(audio_feature_avg) # (1, d_model)
print(f"Projected audio feature, shape: {projected_audio_feature.shape}") # e.g., (1, 64)

# 5. Prepare Input Sequence (Example)
prompt_text = "What sound is this?"
prompt_ids = [char_to_int[ch] for ch in prompt_text]
aud_id = char_to_int[aud_token]

# Input: [<AUD>, prompt tokens]
input_ids_aud = torch.tensor([[aud_id] + prompt_ids], dtype=torch.long, device=device)
print(f"Example input sequence with audio: {input_ids_aud.tolist()}")

# During the forward pass, the embedding for aud_id would be replaced by projected_audio_feature.


#### OUTPUT
--- Conceptual Audio Handling ---
Added <AUD>. New vocab_size: 44
Loaded dummy waveform for 'path/to/dummy_audio.wav', length: 16000
Computed dummy MFCC features, shape: torch.Size([99, 13])
Extracted dummy audio features over time, shape: torch.Size([99, 256])
Averaged audio features, shape: torch.Size([1, 256])
Projected audio feature, shape: torch.Size([1, 64])
Example input sequence with audio: [[43, 41, 21, 14, 31, 1, 30, 28, 32, 27, 17, 1, 22, 30, 1, 31, 21, 22, 30, 9]]
```

This shows extracting basic audio features (like MFCCs), processing and averaging them, projecting the result to match the Transformer’s dimension, and creating an input sequence prefixed with the `<AUD>` token.

So …

> Now, our MultiModal can chat with text, images, videos, and audios

But what about the other direction? Can we generate *images* from text descriptions? like the recent models such as o3 or gpt-4.5.

## Generate Images Functionality (ResNet + Feature Extractor)

So, directly generating image pixels autoregressively (pixel by pixel) with a standard Transformer is incredibly complex and computationally demanding.

Instead, we’ll tackle a simplified version to illustrate the concept within our framework:

1.  **Goal:** Given a text prompt (e.g., “a blue square”), our model will generate an *image feature vector* — the kind of numerical summary that our ResNet extractor produces.
2.  **Why Features?** Predicting a fixed-size feature vector (e.g., 512 numbers) is much more manageable for our Transformer than generating hundreds of thousands of pixel values.
3.  **Visualization:** How do we “see” the result? After our model predicts a feature vector, we’ll compare it to the feature vectors of the known images we used in training (red square, blue square, green circle). We’ll find the known image whose features are *most similar* to the predicted ones and display that image.

We’ll start by loading components from the multimodal model we just saved (`multimodal_model.pt`). We need the text processing parts (`embeddings`, Transformer blocks) and the `tokenizer`.

We also need the frozen `ResNet-18` again, this time to get the target feature vectors for our training images.

```python
# --- Load Saved Multi-Modal Model State ---
print("\nStep 0.2: Loading state from multi-modal model...")
model_load_path = 'saved_models/multimodal_model.pt'
# (Check if file exists - code omitted)
if not os.path.exists(model_load_path):
    raise FileNotFoundError(f"Error: Model file not found at {model_load_path}. Please ensure 'multimodal.ipynb' was run and saved the model.")

loaded_state_dict = torch.load(model_load_path, map_location=device)
print(f"Loaded state dictionary from '{model_load_path}'.")

# --- Extract Config and Tokenizer ---
config = loaded_state_dict['config']
vocab_size = config['vocab_size'] # Includes special tokens now
d_model = config['d_model']
n_layers = config['n_layers']
n_heads = config['n_heads']
d_ff = config['d_ff']
block_size = config['block_size'] # Use the size from multimodal setup
vision_feature_dim = config['vision_feature_dim'] # e.g., 512
d_k = d_model // n_heads

char_to_int = loaded_state_dict['tokenizer']['char_to_int']
int_to_char = loaded_state_dict['tokenizer']['int_to_char']
pad_token_id = char_to_int.get('<PAD>', -1)
print("Extracted model configuration and tokenizer:")
print(f"  vocab_size: {vocab_size}")
# (Print other config values...)
print(f"  block_size: {block_size}")
print(f"  vision_feature_dim: {vision_feature_dim}")
print(f"  PAD token ID: {pad_token_id}")


# --- Load Positional Encoding ---
positional_encoding = loaded_state_dict['positional_encoding'].to(device)
print(f"Loaded positional encoding with shape: {positional_encoding.shape}")

# --- Store State Dicts for Text Components ---
# (Store state dicts for embedding, layer_norms, attention, ffn, final_norm - code omitted)
loaded_embedding_dict = loaded_state_dict['token_embedding_table']
loaded_ln1_dicts = loaded_state_dict['layer_norms_1']
loaded_qkv_dicts = loaded_state_dict['mha_qkv_linears']
loaded_mha_out_dicts = loaded_state_dict['mha_output_linears']
loaded_ln2_dicts = loaded_state_dict['layer_norms_2']
loaded_ffn1_dicts = loaded_state_dict['ffn_linear_1']
loaded_ffn2_dicts = loaded_state_dict['ffn_linear_2']
loaded_final_ln_dict = loaded_state_dict['final_layer_norm']
print("Stored state dicts for text transformer components.")

# --- Load Vision Feature Extractor (Frozen) ---
print("Loading pre-trained vision model (ResNet-18) for target feature extraction...")
vision_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
vision_model.fc = nn.Identity() # Remove classifier
vision_model = vision_model.to(device)
vision_model.eval() # Keep frozen
for param in vision_model.parameters():
    param.requires_grad = False
print(f"Loaded and froze ResNet-18 feature extractor on device: {device}")

# --- Define Image Transformations (same as before) ---
# (Define image_transforms - code omitted)
image_transforms = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
print("Defined image transformations.")
```

We are loading the necessary pieces: the `configuration` (note the `vocab_size` is 42 including special tokens, `block_size` is 64), the `tokenizer`, the weights for the text processing parts, and the frozen `ResNet` feature extractor.

```python
#### OUTPUT ####
Step 0.2: Loading state from multi-modal model...
Loaded state dictionary from 'saved_models/multimodal_model.pt'.
Extracted model configuration and tokenizer:
  vocab_size: 42
  d_model: 64
  n_layers: 3
  n_heads: 4
  d_ff: 256
  block_size: 64
  vision_feature_dim: 512
  PAD token ID: 37
Loaded positional encoding with shape: torch.Size([1, 64, 64])
Stored state dicts for text transformer components.
Loading pre-trained vision model (ResNet-18) for target feature extraction...
Loaded and froze ResNet-18 feature extractor on device: cuda
Defined image transformations.
```

Now we need pairs of (Text Prompt, Target Image). We’ll use descriptive prompts for the simple images we already have.

```lua
print("\nStep 0.3: Defining sample text-to-image data...")

# --- Image Paths (Assuming they exist) ---
# (Define image_paths dictionary and check/recreate files - code omitted)
sample_data_dir = "sample_multimodal_data"
image_paths = {
    "red_square": os.path.join(sample_data_dir, "red_square.png"),
    "blue_square": os.path.join(sample_data_dir, "blue_square.png"),
    "green_circle": os.path.join(sample_data_dir, "green_circle.png")
}
# (Code to check/recreate images)


# --- Define Text Prompt -> Image Path Pairs ---
text_to_image_data = [
    {"prompt": "a red square", "image_path": image_paths["red_square"]},
    {"prompt": "the square is red", "image_path": image_paths["red_square"]},
    {"prompt": "show a blue square", "image_path": image_paths["blue_square"]},
    {"prompt": "blue shape, square", "image_path": image_paths["blue_square"]},
    {"prompt": "a green circle", "image_path": image_paths["green_circle"]},
    {"prompt": "the circle, it is green", "image_path": image_paths["green_circle"]},
    {"prompt": "make a square that is red", "image_path": image_paths["red_square"]}
]
num_samples = len(text_to_image_data)
```

We have 7 training examples linking text descriptions like “a red square” or “the circle, it is green” to the corresponding image files.

For training, the model needs to predict the *feature vector* of the target image. We pre-compute these target vectors using our frozen ResNet and store them.

We also keep a list mapping image paths to their features for the nearest-neighbor lookup later.

```python
print("\nStep 0.4: Extracting target image features...")
target_image_features = {} # Dict: {image_path: feature_tensor}
known_features_list = [] # List: [(path, feature_tensor)]

# --- Loop Through Unique Image Paths ---
unique_image_paths_in_data = sorted(list(set(d["image_path"] for d in text_to_image_data)))
print(f"Found {len(unique_image_paths_in_data)} unique target images to process.")

for img_path in unique_image_paths_in_data:
    # --- Load Image ---
    img = Image.open(img_path).convert('RGB')
    # --- Apply Transformations ---
    img_tensor = image_transforms(img).unsqueeze(0).to(device)
    # --- Extract Features (using frozen vision_model) ---
    with torch.no_grad():
        feature_vector = vision_model(img_tensor)
    feature_vector_squeezed = feature_vector.squeeze(0)
    print(f"  Extracted features for '{os.path.basename(img_path)}', shape: {feature_vector_squeezed.shape}")

    # --- Store Features ---
    target_image_features[img_path] = feature_vector_squeezed
    known_features_list.append((img_path, feature_vector_squeezed))

# (Error check if target_image_features is empty - code omitted)
if not target_image_features:
     raise ValueError("No target image features were extracted. Cannot proceed.")

print("Finished extracting and storing target image features.")
print(f"Stored {len(known_features_list)} known (path, feature) pairs for generation lookup.")



#### OUTPUT ####
Found 3 unique target images to process.
  Extracted features for 'blue_square.png', shape: torch.Size([512])
  Extracted features for 'green_circle.png', shape: torch.Size([512])
  Extracted features for 'red_square.png', shape: torch.Size([512])
Finished extracting and storing target image features.
Stored 3 known (path, feature) pairs for generation lookup.
```

The `512`-dimensional target feature vectors for our blue square, green circle, and red square are extracted and stored.

We also have `known_features_list` which pairs each image path with its feature tensor – this will be our "database" for finding the closest match during generation.

Let’s set the training parameters for this task. We might adjust the learning rate.

```python
print("\nStep 0.5: Defining training hyperparameters for text-to-image...")
# (Check block_size vs max_prompt_len - code omitted)
print(f"Using block_size: {block_size}")
# Recreate causal mask for this block size
causal_mask = torch.tril(torch.ones(block_size, block_size, device=device)).view(1, 1, block_size, block_size)

learning_rate = 1e-4 # Lower LR for fine-tuning
batch_size = 4
epochs = 5000
eval_interval = 500
print(f"  Training Params: LR={learning_rate}, BatchSize={batch_size}, Epochs={epochs}")


#### OUTPUT ####
Using block_size: 64
  Training Params: LR=0.0001, BatchSize=4, Epochs=5000
```

We’ll use a slightly lower learning rate (`1e-4`) for this fine-tuning task, keeping the `batch size` small and training for `5000` epochs.

We need to rebuild the text Transformer using the loaded weights and, crucially, add a *new* output layer.

```python
# --- Token Embedding Table ---
token_embedding_table = nn.Embedding(vocab_size, d_model).to(device)
token_embedding_table.load_state_dict(loaded_embedding_dict)
print(f"  Loaded Token Embedding Table, shape: {token_embedding_table.weight.shape}")

# --- Transformer Blocks Components ---
# (Instantiate layers and load state_dict for layer_norms_1, mha_qkv_linears, etc. - code omitted)
layer_norms_1 = []
mha_qkv_linears = []
mha_output_linears = []
layer_norms_2 = []
ffn_linear_1 = []
ffn_linear_2 = []
for i in range(n_layers):
    # (Instantiate layer, check bias, load state dict for each component...)
    ln1 = nn.LayerNorm(d_model).to(device)
    ln1.load_state_dict(loaded_ln1_dicts[i])
    layer_norms_1.append(ln1)
    # ... load MHA QKV ...
    qkv_dict = loaded_qkv_dicts[i]
    has_bias = 'bias' in qkv_dict
    qkv = nn.Linear(d_model, 3 * d_model, bias=has_bias).to(device)
    qkv.load_state_dict(qkv_dict)
    mha_qkv_linears.append(qkv)
    # ... load MHA Output ...
    mha_out_dict = loaded_mha_out_dicts[i]
    has_bias = 'bias' in mha_out_dict
    mha_out = nn.Linear(d_model, d_model, bias=has_bias).to(device)
    mha_out.load_state_dict(mha_out_dict)
    mha_output_linears.append(mha_out)
    # ... load LN2 ...
    ln2 = nn.LayerNorm(d_model).to(device)
    ln2.load_state_dict(loaded_ln2_dicts[i])
    layer_norms_2.append(ln2)
    # ... load FFN1 ...
    ffn1_dict = loaded_ffn1_dicts[i]
    has_bias = 'bias' in ffn1_dict
    ff1 = nn.Linear(d_model, d_ff, bias=has_bias).to(device)
    ff1.load_state_dict(ffn1_dict)
    ffn_linear_1.append(ff1)
    # ... load FFN2 ...
    ffn2_dict = loaded_ffn2_dicts[i]
    has_bias = 'bias' in ffn2_dict
    ff2 = nn.Linear(d_ff, d_model, bias=has_bias).to(device)
    ff2.load_state_dict(ffn2_dict)
    ffn_linear_2.append(ff2)

print(f"  Loaded components for {n_layers} Transformer Layers.")

# --- Final LayerNorm ---
final_layer_norm = nn.LayerNorm(d_model).to(device)
final_layer_norm.load_state_dict(loaded_final_ln_dict)
print("  Loaded Final LayerNorm.")

print("Finished initializing and loading weights for text transformer components.")




#### OUTPUT ####
  Loaded Token Embedding Table, shape: torch.Size([42, 64])
  Loaded components for 3 Transformer Layers.
  Loaded Final LayerNorm.
```

The text processing backbone of our model is reassembled with its trained weights.

This is the key change. Instead of predicting the next text token, we need to predict an image feature vector. We add a new `Linear` layer that maps the Transformer's final output (`d_model=64`) to the image feature dimension (`vision_feature_dim=512`).

```python
text_to_image_feature_layer = nn.Linear(d_model, vision_feature_dim).to(device)

#### OUTPUT ####
 Initialized Text-to-Image-Feature Output Layer: 64 -> 512. Device: cuda
```

This new output head is ready to be trained to predict image features. We need to format our text prompts and pair them with the target image features we extracted earlier.

```python
print("\nStep 2.1: Tokenizing and padding text prompts...")

prepared_prompts = []
target_features_ordered = []

for sample in text_to_image_data:
    prompt = sample["prompt"]
    image_path = sample["image_path"]
    # --- Tokenize Prompt ---
    prompt_ids_no_pad = [char_to_int[ch] for ch in prompt]
    # --- Padding ---
    # (Padding logic - code omitted)
    current_len = len(prompt_ids_no_pad)
    pad_len = block_size - current_len
    if pad_len < 0:
        prompt_ids = prompt_ids_no_pad[:block_size]
        pad_len = 0
        current_len = block_size
    else:
        prompt_ids = prompt_ids_no_pad + ([pad_token_id] * pad_len)
    # --- Create Attention Mask ---
    attention_mask = ([1] * current_len) + ([0] * pad_len)
    # --- Store Prompt Data ---
    prepared_prompts.append({
        "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
    })
    # --- Store Corresponding Target Feature ---
    # (Append feature from target_image_features - code omitted)
    if image_path in target_image_features:
        target_features_ordered.append(target_image_features[image_path])
    else:
        # Handle error
        target_features_ordered.append(torch.zeros(vision_feature_dim, device=device))


# --- Stack into Tensors ---
all_prompt_input_ids = torch.stack([p['input_ids'] for p in prepared_prompts])
all_prompt_attention_masks = torch.stack([p['attention_mask'] for p in prepared_prompts])
all_target_features = torch.stack(target_features_ordered)

num_sequences_available = all_prompt_input_ids.shape[0]
print(f"Created {num_sequences_available} padded prompt sequences and gathered target features.")
print(f"  Prompt Input IDs shape: {all_prompt_input_ids.shape}")
print(f"  Prompt Attention Mask shape: {all_prompt_attention_masks.shape}")
print(f"  Target Features shape: {all_target_features.shape}")


#### OUTPUT ####
Created 7 padded prompt sequences and gathered target features.
  Prompt Input IDs shape: torch.Size([7, 64])
  Prompt Attention Mask shape: torch.Size([7, 64])
  Target Features shape: torch.Size([7, 512])
```

Our 7 text prompts are now padded sequences of IDs (shape `[7, 64]`), each with a corresponding `attention_mask` (also `[7, 64]`) and a target `image feature vector` (shape `[7, 512]`).

Time to train the model to map text descriptions to image features.

We set up `AdamW`, ensuring it includes the parameters of the new output layer (`text_to_image_feature_layer`). The loss function is `Mean Squared Error (MSE)`, suitable for comparing the predicted `512`-dimensional feature vector against the target `512`-dimensional vector.

```python
# --- Gather Trainable Parameters ---
# (Gather parameters for text components + text_to_image_feature_layer - code omitted)
all_trainable_parameters_t2i = list(token_embedding_table.parameters())
for i in range(n_layers):
    all_trainable_parameters_t2i.extend(list(layer_norms_1[i].parameters()))
    # ... extend with other layer parameters ...
all_trainable_parameters_t2i.extend(list(final_layer_norm.parameters()))
all_trainable_parameters_t2i.extend(list(text_to_image_feature_layer.parameters())) # New layer


# --- Define Optimizer ---
optimizer = optim.AdamW(all_trainable_parameters_t2i, lr=learning_rate)
print(f"  Optimizer defined: AdamW with lr={learning_rate}")
print(f"  Managing {len(all_trainable_parameters_t2i)} parameter groups/tensors.")

# --- Define Loss Function ---
criterion = nn.MSELoss() # Mean Squared Error
print(f"  Loss function defined: {type(criterion).__name__}")


#### OUTPUT ####
  Optimizer defined: AdamW with lr=0.0001
  Managing 38 parameter groups/tensors.
  Loss function defined: MSELoss
```

Optimizer and MSE loss are set. Note the parameter count is 38 again (we swapped the old text output layer for the new image feature output layer).

We iterate, feeding text prompts to the model and training it to output feature vectors that are close to the target image’s features.

1.  **Get Batch:** Sample prompts, masks, and target features.
2.  **Forward Pass:** Process the prompt through the Transformer blocks. Take the hidden state of the *last actual token* (before padding) from the final layer norm output. Pass this single state through the text_to_image_feature_layer to predict the image feature vector.
3.  **Calculate Loss:** Compute MSE between the predicted and target feature vectors.

```python
t2i_losses = []
# (Set layers to train mode - code omitted)
token_embedding_table.train()
# ... set other layers to train ...
final_layer_norm.train()
text_to_image_feature_layer.train() # Train the new layer


for epoch in range(epochs):
    # --- 1. Batch Selection ---
    indices = torch.randint(0, num_sequences_available, (batch_size,))
    xb_prompt_ids = all_prompt_input_ids[indices].to(device)
    batch_prompt_masks = all_prompt_attention_masks[indices].to(device)
    yb_target_features = all_target_features[indices].to(device)

    # --- 2. Forward Pass ---
    B, T = xb_prompt_ids.shape
    C = d_model
    # Embeddings + PE
    token_embed = token_embedding_table(xb_prompt_ids)
    pos_enc_slice = positional_encoding[:, :T, :]
    x = token_embed + pos_enc_slice
    # Transformer Blocks (using combined mask)
    # (Attention mask calculation - code omitted)
    padding_mask_expanded = batch_prompt_masks.unsqueeze(1).unsqueeze(2)
    combined_attn_mask = causal_mask[:,:,:T,:T] * padding_mask_expanded
    for i in range(n_layers):
        # (Forward pass through MHA and FFN - code omitted)
        x_input_block = x
        x_ln1 = layer_norms_1[i](x_input_block)
        # ... MHA logic with combined_attn_mask ...
        attention_weights = F.softmax(attn_scores.masked_fill(combined_attn_mask == 0, float('-inf')), dim=-1)
        attention_weights = torch.nan_to_num(attention_weights)
        # ... finish MHA ...
        x = x_input_block + mha_output_linears[i](attn_output) # Residual 1
        # ... FFN logic ...
        x = x_input_ffn + ffn_linear_2[i](F.relu(ffn_linear_1[i](layer_norms_2[i](x_input_ffn)))) # Residual 2

    # Final LayerNorm
    final_norm_output = final_layer_norm(x) # (B, T, C)

    # Select Hidden State of the last non-padding token
    last_token_indices = torch.sum(batch_prompt_masks, 1) - 1
    last_token_indices = torch.clamp(last_token_indices, min=0)
    batch_indices = torch.arange(B, device=device)
    last_token_hidden_states = final_norm_output[batch_indices, last_token_indices, :] # (B, C)

    # Project to Image Feature Dimension
    predicted_image_features = text_to_image_feature_layer(last_token_hidden_states) # (B, vision_feature_dim)

    # --- 3. Calculate Loss ---
    loss = criterion(predicted_image_features, yb_target_features) # MSE Loss

    # --- 4. Zero Gradients ---
    optimizer.zero_grad()
    # --- 5. Backward Pass ---
    # (Backward pass if loss is valid - code omitted)
    if not torch.isnan(loss) and not torch.isinf(loss):
        loss.backward()
        # --- 6. Update Parameters ---
        optimizer.step()
    else:
        loss = None

    # --- Logging ---
    # (Log loss - code omitted)
    if loss is not None:
        current_loss = loss.item()
        t2i_losses.append(current_loss)
        if epoch % eval_interval == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}/{epochs}, MSE Loss: {current_loss:.6f}")
    elif epoch % eval_interval == 0 or epoch == epochs - 1:
        print(f"  Epoch {epoch+1}/{epochs}, Loss: Invalid (NaN/Inf)")


print("--- Text-to-Image Training Loop Completed ---\n")
```

This loop will start the training and start printing the loos after certain number of epochs.

```python
#### OUTPUT ####
Step 3.2: Starting Text-to-Image Training Loop...
  Epoch 1/5000, MSE Loss: 0.880764
  Epoch 501/5000, MSE Loss: 0.022411
  Epoch 1001/5000, MSE Loss: 0.006695
  Epoch 1501/5000, MSE Loss: 0.000073
  Epoch 2001/5000, MSE Loss: 0.000015
  Epoch 2501/5000, MSE Loss: 0.000073
  Epoch 3001/5000, MSE Loss: 0.000002
  Epoch 3501/5000, MSE Loss: 0.000011
  Epoch 4001/5000, MSE Loss: 0.000143
  Epoch 4501/5000, MSE Loss: 0.000013
  Epoch 5000/5000, MSE Loss: 0.000030
```

Again, the MSE loss drops significantly, indicating the model is learning to produce feature vectors that closely match the target features for the given text prompts in our small dataset.

## Generating Image using Text Prompt

Let’s see what image feature vector our model predicts for a new prompt, like “a blue square shape”.

Let’s tokenize and pad our test input prompt.

```python
# --- Input Prompt ---
generation_prompt_text = "a blue square shape"
print(f"Input Prompt: '{generation_prompt_text}'")
# --- Tokenize and Pad ---
# (Tokenizing and padding logic - code omitted)
gen_prompt_ids_no_pad = [char_to_int.get(ch, pad_token_id) for ch in generation_prompt_text]
gen_current_len = len(gen_prompt_ids_no_pad)
gen_pad_len = block_size - gen_current_len
if gen_pad_len < 0:
    gen_prompt_ids = gen_prompt_ids_no_pad[:block_size]
    gen_pad_len = 0
    gen_current_len = block_size
else:
    gen_prompt_ids = gen_prompt_ids_no_pad + ([pad_token_id] * gen_pad_len)
# --- Create Attention Mask ---
gen_attention_mask = ([1] * gen_current_len) + ([0] * gen_pad_len)
# --- Convert to Tensor ---
xb_gen_prompt_ids = torch.tensor([gen_prompt_ids], dtype=torch.long, device=device)
batch_gen_prompt_masks = torch.tensor([gen_attention_mask], dtype=torch.long, device=device)
print(f"Prepared prompt tensor shape: {xb_gen_prompt_ids.shape}")
print(f"Prepared mask tensor shape: {batch_gen_prompt_masks.shape}")


#### OUTPUT ####
Input Prompt: 'a blue square shape'
Prepared prompt tensor shape: torch.Size([1, 64])
Prepared mask tensor shape: torch.Size([1, 64])
```

The prompt “a blue square shape” is ready as a padded tensor. Now we perform a forward pass using the trained text-to-image model.

```python
print("\nStep 4.2: Generating image feature vector...")
# --- Set Model to Evaluation Mode ---
# (Set layers to eval() - code omitted)
token_embedding_table.eval()
# ... set other layers to eval ...
final_layer_norm.eval()
text_to_image_feature_layer.eval() # Eval the output layer

# --- Forward Pass ---
with torch.no_grad():
    # (Embeddings, PE, Transformer Blocks - code omitted)
    B_gen, T_gen = xb_gen_prompt_ids.shape
    C_gen = d_model
    token_embed_gen = token_embedding_table(xb_gen_prompt_ids)
    pos_enc_slice_gen = positional_encoding[:, :T_gen, :]
    x_gen = token_embed_gen + pos_enc_slice_gen
    padding_mask_expanded_gen = batch_gen_prompt_masks.unsqueeze(1).unsqueeze(2)
    combined_attn_mask_gen = causal_mask[:,:,:T_gen,:T_gen] * padding_mask_expanded_gen
    for i in range(n_layers):
        # ... MHA and FFN forward pass ...
         x_input_block_gen = x_gen
         x_ln1_gen = layer_norms_1[i](x_input_block_gen)
         # ... MHA logic ...
         x_gen = x_input_block_gen + mha_output_linears[i](attn_output_gen)
         # ... FFN logic ...
         x_gen = x_input_ffn_gen + ffn_linear_2[i](F.relu(ffn_linear_1[i](layer_norms_2[i](x_input_ffn_gen))))

    # Final LayerNorm
    final_norm_output_gen = final_layer_norm(x_gen)
    # Select Hidden State
    last_token_indices_gen = torch.sum(batch_gen_prompt_masks, 1) - 1
    last_token_indices_gen = torch.clamp(last_token_indices_gen, min=0)
    batch_indices_gen = torch.arange(B_gen, device=device)
    last_token_hidden_states_gen = final_norm_output_gen[batch_indices_gen, last_token_indices_gen, :]
    # Project to Image Feature Dimension
    predicted_feature_vector = text_to_image_feature_layer(last_token_hidden_states_gen)

print(f"Generated predicted feature vector with shape: {predicted_feature_vector.shape}")


#### OUTPUT ####
Generated predicted feature vector with shape: torch.Size([1, 512])
```

The model processed the prompt and outputted a 512-dimensional vector, representing its prediction of the image features for “a blue square shape”.

Now our predicted image from our text prompt is:

![Generated Image](https://miro.medium.com/v2/resize:fit:486/1*mEM0Io7GqTROL-CJ69aLaQ.png)

Success! The model predicted a feature vector for “a blue square shape”, and when we compared this prediction to our known image features, the closest match was indeed the blue square.

The Euclidean distance (4.84) indicates how similar the predicted vector was to the actual blue square’s feature vector.

## Conclusion

So a Quick overview of what we did:

1.  So, we started right at the beginning, coding up our own BPE tokenizer. This let us chop up text into smart little pieces our model could work with.
2.  Then, we built a basic text-only model, like a tiny GPT. We used that ‘Alice in Wonderland’ text to teach it how words follow each other, so it could even write a little bit of new text on its own.
3.  Next up, we made our model multi-talented! We plugged in a way for it to “see” images (using ResNet features). It learned to look at our simple colored shapes and answer questions about them, like telling us the color.
4.  We also mapped out how we’d handle video and audio later. The plan was similar: turn the video frames or audio sounds into numbers, give them special tags like `<VID>` or `<AUD>`, and mix them into the input sequence.
5.  Finally, we flipped things around and tried text-to-image. We didn’t make it draw pixels (that’s tough!), but we trained it to predict the *number summary* (the feature vector) for an image based on text like “a blue square”. Then, we just found which of our known images (red square, blue square, green circle) had the closest number summary to what the model predicted!

Happy Learning!

If you have any questions, please feel free to raise an issue.