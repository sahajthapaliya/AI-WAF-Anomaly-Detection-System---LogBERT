# WAF Anomaly Detection using LogBERT

This directory contains the implementation of LogBERT for detecting anomalies in Web Application Firewall (WAF) command inputs. The implementation is based on the LogBERT framework developed by Helen Guoh and her team. https://github.com/HelenGuohx/logbert.git. If needed to clone, go to this link and follow the instructions to get the base framework. I have uploaded only the files that are different to the original framework. Replace predict_log.py and train_log.py after the environment is installed. The directory to replace the files is Your Drive:\Your Project Folder\env\Lib\site-packages\bert_pytorch\. Make sure to install the necessary packages.

## Overview

The WAF implementation processes raw command inputs from various attack categories, tokenizes them, and trains the LogBERT model to detect anomalies. The implementation also includes baseline models like DeepLog and LogAnomaly for comparison.

## Data Structure

The raw command inputs are located in the `../data/` directory, organized by attack categories:
- cmdexe
- log4shell
- shellshock
- sqli
- traversal
- xss
- xxe

Each category folder contains:
- `normal_*.txt`: Normal command examples
- `attack_*.txt`: Attack command examples

## Usage

### 1. Initialize the Environment

First, run the initialization script to create the necessary directories:

```bash
sh init.sh
```

### 2. Process Data

Process the raw command inputs to prepare them for the models:

```bash
python data_process.py
```

This will:
- Read all normal and attack commands
- Tokenize them
- Create a vocabulary
- Split the data into training and testing sets

### 3. Train and Test Models

#### LogBERT

```bash
# Generate vocabulary
python logbert.py vocab

# Train the model
python logbert.py train

# Test the model
python logbert.py predict
```


## Implementation Details

### Data Processing

The `data_process.py` script:
1. Reads raw command inputs from all attack categories
2. Tokenizes them by splitting on special characters
3. Creates a vocabulary mapping tokens to IDs
4. Splits the data into training (normal commands) and testing (normal and attack commands)

### Model Configuration

The models are configured with parameters optimized for the WAF command input task:

- **LogBERT**: Uses a window size of 64, sequence length of 256, and minimum length of 5


## Evaluation

The models are evaluated on their ability to:
1. Correctly identify normal commands (false positive rate)
2. Detect attack commands (detection rate)

Results are stored in the `../output/waf/` directory. 

I worked with highly imbalanced dataset. If you can find/generate a balanced dataset, then I recommend that.
