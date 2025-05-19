import sys
sys.path.append('../')

import os
import re
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import glob

# Directory setup
data_dir = 'F:/logbert-main/data2/'
output_dir = 'F:/logbert-main/WAF/output/'

# Attack categories
attack_categories = ['shellshock2','cmdexe2', 'sqli2','xss2','traversal2','log4shell2','xxe2']

def process_raw_commands():
    """
    Process raw command inputs from various attack categories and convert them
    to a structured format that can be used by logBERT.
    """
    print("Processing raw command inputs...")
    
    # Create a DataFrame to store all commands with their labels
    all_data = []
    
    # Process each attack category
    for category in attack_categories:
        print(f"Processing {category}...")
        
        # Read normal commands
        normal_file = os.path.join(data_dir, category, f"normal_{category}.txt")
        if os.path.exists(normal_file):
            with open(normal_file, 'r', encoding='utf-8', errors='ignore') as f:
                normal_commands = f.readlines()
            
            for cmd in normal_commands:
                cmd = cmd.strip()
                if cmd:  # Skip empty lines
                    all_data.append({
                        'Content': cmd,
                        'Category': category,
                        'Label': 'Normal'
                    })
        
        # Read attack commands
        attack_file = os.path.join(data_dir, category, f"attack_{category}.txt")
        if os.path.exists(attack_file):
            with open(attack_file, 'r', encoding='utf-8', errors='ignore') as f:
                attack_commands = f.readlines()
            
            # Track seen commands to avoid duplicates
            seen_commands = set()
            for cmd in attack_commands:
                cmd = cmd.strip()
                if cmd and cmd not in seen_commands:  # Skip empty lines and duplicates
                    seen_commands.add(cmd)
                    all_data.append({
                        'Content': cmd,
                        'Category': category,
                        'Label': 'Attack'
                    })
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save the structured data
    structured_file = os.path.join(output_dir, "waf_structured.csv")
    df.to_csv(structured_file, index=False)
    print(f"Saved structured data to {structured_file}")
    
    return df

def tokenize_commands(df):
    """
    Tokenize command inputs to prepare them for LogBERT.
    """
    print("Tokenizing commands...")
    
    # Simple tokenization - split by special characters and keep them as tokens
    def tokenize(cmd):
        # Replace common special characters with spaces around them for tokenization
        for char in "[](){}<>=!@#$%^&*-+;:,./?|\"\\'":
            cmd = cmd.replace(char, f" {char} ")
        # Split by whitespace and filter out empty tokens
        tokens = [token for token in cmd.split() if token]
        return tokens
    
    df['Tokens'] = df['Content'].apply(tokenize)
    
    # Create a vocabulary from all tokens
    all_tokens = set()
    for tokens in df['Tokens']:
        all_tokens.update(tokens)
    
    # Assign a unique ID to each token
    token_to_id = {token: idx+1 for idx, token in enumerate(sorted(all_tokens))}
    
    # Save token mapping
    with open(os.path.join(output_dir, "waf_token_mapping.json"), 'w') as f:
        json.dump(token_to_id, f)
    
    # Convert tokens to IDs
    df['EventSequence'] = df['Tokens'].apply(lambda tokens: [token_to_id[token] for token in tokens])
    
    return df, token_to_id

def generate_train_test_data(df, ratio=0.7):
    """
    Generate training and testing datasets.
    Uses stratified sampling to ensure similar distributions of normal traffic patterns 
    in both training and test sets.
    """
    print("Generating train/test datasets...")
    
    # Separate normal and attack data
    normal_df = df[df['Label'] == 'Normal']
    attack_df = df[df['Label'] == 'Attack']
    
    # Perform stratified sampling for normal data by category
    train_normal_parts = []
    test_normal_parts = []
    
    # For each category, split the normal data
    for category in normal_df['Category'].unique():
        category_df = normal_df[normal_df['Category'] == category]
        # Shuffle with fixed seed for reproducibility
        category_df = category_df.sample(frac=1, random_state=42)
        
        # Split this category's data
        train_size = int(len(category_df) * ratio)
        train_normal_parts.append(category_df.iloc[:train_size])
        test_normal_parts.append(category_df.iloc[train_size:])
        
        print(f"Category {category}: {len(category_df)} samples split into {train_size} train and {len(category_df) - train_size} test")
    
    # Combine all parts
    train_normal = pd.concat(train_normal_parts)
    test_normal = pd.concat(test_normal_parts)
    
    # Shuffle again to mix categories
    train_normal = train_normal.sample(frac=1, random_state=42)
    test_normal = test_normal.sample(frac=1, random_state=42)
    
    # Shuffle attack data
    test_abnormal = attack_df.sample(frac=1, random_state=42)
    
    # Save sequences to files
    df_to_file(train_normal['EventSequence'], os.path.join(output_dir, "train"))
    df_to_file(test_normal['EventSequence'], os.path.join(output_dir, "test_normal"))
    df_to_file(test_abnormal['EventSequence'], os.path.join(output_dir, "test_abnormal"))
    
    # Save the data splits for analysis
    train_normal.to_csv(os.path.join(output_dir, "train_normal_data.csv"), index=False)
    test_normal.to_csv(os.path.join(output_dir, "test_normal_data.csv"), index=False)
    test_abnormal.to_csv(os.path.join(output_dir, "test_abnormal_data.csv"), index=False)
    
    print(f"\nSummary:")
    print(f"Normal data: {len(normal_df)}, Attack data: {len(attack_df)}")
    print(f"Training data: {len(train_normal)}, Normal test data: {len(test_normal)}, Attack test data: {len(test_abnormal)}")
    
    # Print detailed distribution information
    print("\nCategory distribution in training data:")
    print(train_normal['Category'].value_counts())
    
    print("\nCategory distribution in normal test data:")
    print(test_normal['Category'].value_counts())
    
    print("\nCategory distribution in attack test data:")
    print(test_abnormal['Category'].value_counts())

def df_to_file(event_sequences, file_name):
    """
    Write event sequences to a file.
    """
    with open(file_name, 'w') as f:
        for seq in event_sequences:
            f.write(' '.join([str(ele) for ele in seq]))
            f.write('\n')

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process raw commands
    df = process_raw_commands()
    
    # Tokenize commands
    df, token_mapping = tokenize_commands(df)
    
    # Generate train/test data
    generate_train_test_data(df)
    
    print("Data processing complete!") 