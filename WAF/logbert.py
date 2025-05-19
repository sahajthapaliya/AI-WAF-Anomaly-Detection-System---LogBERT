import sys
#sys.path.append("../")
#sys.path.append("../../")

import os
#dirname = os.path.dirname(__file__)

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import random
import pickle

from bert_pytorch.dataset import WordVocab, LogDataset
from bert_pytorch.model import BERT
from bert_pytorch.trainer import BERTTrainer
from bert_pytorch.predict_log import Predictor

from bert_pytorch import Trainer
from bert_pytorch.dataset.utils import seed_everything
# Get absolute project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Set up paths with absolute references
options = dict()
options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
options["output_dir"] = os.path.join(PROJECT_ROOT, "WAF","output") + os.sep
options["model_dir"] = os.path.join(options["output_dir"], "bert")
options["model_path"] = os.path.join(options["model_dir"], "best_bert.pth")
options["train_vocab"] = os.path.join(options["output_dir"], "train")  
options["vocab_path"] = os.path.join(options["output_dir"], "vocab.pkl")
options["scale_path"] = os.path.join(options["model_dir"], "scale.pkl")

options["window_size"] = 64  # Adjusted for potentially shorter command sequences
options["adaptive_window"] = True
options["seq_len"] = 256  # Adjusted for command inputs
options["max_len"] = 256  # for position embedding
options["min_len"] = 3 # Adjusted for shorter command sequences
options["mask_ratio"] = 0.37 # Optimal masking for anomaly detection
# sample ratio
options["train_ratio"] = 1
options["valid_ratio"] = 0.3
options["test_ratio"] = 1

# features
options["is_logkey"] = True
options["is_time"] = False

options["hypersphere_loss"] = True
options["hypersphere_loss_test"] = False


options["scale"] = None # MinMaxScaler()

# model
options["hidden"] = 128 # embedding size
options["layers"] = 2
options["attn_heads"] = 4

options["epochs"] = 10
options["n_epochs_stop"] = 10
options["batch_size"] = 32

options["corpus_lines"] = None
options["on_memory"] = True
options["num_workers"] = 5
options["lr"] = 1e-3
options["adam_beta1"] = 0.9
options["adam_beta2"] = 0.999
options["adam_weight_decay"] = 0.00
options["with_cuda"]= True
options["cuda_devices"] = None
options["log_freq"] = None

# predict
options["num_candidates"] = 6
options["gaussian_mean"] = 0
options["gaussian_std"] = 1

seed_everything(seed=1234)

if not os.path.exists(options['model_dir']):
    os.makedirs(options['model_dir'], exist_ok=True)

print("device", options["device"])
print("features logkey:{} time: {}\n".format(options["is_logkey"], options["is_time"]))
print("mask ratio", options["mask_ratio"])

class LogBERT:
    def __init__(self):
        self.options = options
        self.model_dir = options["model_dir"]
        self.output_dir = options["output_dir"]
        self.vocab_path = options["vocab_path"]
        self.window_size = options["window_size"]
        self.train_path = options["train_path"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.header = options["header"] if "header" in options else None

    def train(self):
        model_dir = self.model_dir
        options = self.options
        device = options['device']
        num_epochs = options['epoch']
        header = self.header
        
        print("Loading training datasets...")
        # ... rest of the train method ...

    def run(self):
        """Main entry point for running the LogBERT model"""
        
        options = self.options
        model_dir = self.model_dir
        
        print("Starting LogBERT...")
        
        # Training phase
        if options["train"]:
            self.train()
        
        # Prediction phase
        if options["predict"]:
            print("Running anomaly detection...")
            
            # Create predictor
            predictor = Predictor(options=options, 
                                  vocab_path=self.vocab_path,
                                  model_path=model_dir + "checkpoint.bert",
                                  model_dir=model_dir,
                                  output_dir=self.output_dir,
                                  test_normal_sessions=options.get("test_normal_path", self.output_dir + "test_normal"),
                                  test_abnormal_sessions=options.get("test_abnormal_path", self.output_dir + "test_abnormal")
                                 )
            
            predictor.predict()
            
            print("Anomaly detection completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(mode='train')

    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(mode='predict')
    predict_parser.add_argument("-m", "--mean", type=float, default=0)
    predict_parser.add_argument("-s", "--std", type=float, default=1)
    predict_parser.add_argument("-t", "--threshold", default=0.5, type=float,
                           help="threshold for anomaly detection")

    vocab_parser = subparsers.add_parser('vocab')
    vocab_parser.set_defaults(mode='vocab')
    vocab_parser.add_argument("-s", "--vocab_size", type=int, default=None)
    vocab_parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    vocab_parser.add_argument("-m", "--min_freq", type=int, default=1)

    args = parser.parse_args()
    print("arguments", args)

    if args.mode == 'train':
        print("Starting training...")
        options["train_path"] = os.path.join(options["output_dir"], "train")
        Trainer(options).train()

    elif args.mode == 'predict':
        print("Starting prediction...")
        options["gaussian_mean"] = args.mean
        options["gaussian_std"] = args.std
        options["test_normal_path"] = os.path.join(options["output_dir"], "test_normal")
        options["test_abnormal_path"] = os.path.join(options["output_dir"], "test_abnormal")
        options["test_ratio"] = 200 # Limit attack test set to 200 samples

        
        print(f"DEBUG - Normal test file path: {options.get('test_normal_path', 'Not set')}")
        print(f"DEBUG - Abnormal test file path: {options.get('test_abnormal_path', 'Not set')}")
        print(f"DEBUG - Files exist: Normal = {os.path.exists(options.get('test_normal_path', ''))}, Abnormal = {os.path.exists(options.get('test_abnormal_path', ''))}")
        
        Predictor(options).predict()

    elif args.mode == 'vocab':
        print("Building vocabulary...")
        train_file = options["train_vocab"]
        with open(train_file, "r", encoding=args.encoding) as f:
            texts = f.readlines()
        vocab = WordVocab(texts, max_size=args.vocab_size, min_freq=args.min_freq)
        print("VOCAB SIZE:", len(vocab))
        print("save vocab in", options["vocab_path"])
        vocab.save_vocab(options["vocab_path"]) 