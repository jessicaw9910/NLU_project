#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
import transformers
import os
# import time
from utils import annotate_splits, compute_metrics, save_stats, AbstractDataset

def main(args):
    model = args.model.split(':')[0]
    huggingface = args.model.split(':')[1]
    path = args.path + model

    if torch.cuda.is_available():
        torch.device('cuda')
    else:
        torch.device('cpu')

    tokenizer = transformers.AutoTokenizer.from_pretrained(huggingface)

    if model == 'BioMed-RoBERTa':
        model = transformers.RobertaForSequenceClassification.from_pretrained(huggingface, num_labels=2)
    else:
        model = transformers.BertForSequenceClassification.from_pretrained(huggingface, num_labels=2)

    df = pd.read_csv(args.data)

    ## tokenize abstracts
    encoded_train = tokenizer(list(df.loc[df['set'] == 'train', 'abstract']),
                                max_length=args.maxlength, truncation=True, padding=True, return_tensors='pt')
    encoded_val = tokenizer(list(df.loc[df['set'] == 'validation', 'abstract']),
                            max_length=args.maxlength, truncation=True, padding=True, return_tensors='pt')
    # encoded_test = tokenizer(list(df.loc[df['set'] == 'test', 'abstract']),
                            #  max_length=args.maxlength, truncation=True, padding=True, return_tensors='pt')    

    ## create labels
    label_train = list(df.loc[df['set'] == 'train', 'SENSITIVE'])
    label_val = list(df.loc[df['set'] == 'validation', 'SENSITIVE'])
    # label_test = list(df.loc[df['set'] == 'test', 'SENSITIVE'])

    label_train = [int(x == True) for x in label_train]
    label_val = [int(x == True) for x in label_val]
    # label_test = [int(x == True) for x in label_test]

    ## create dataset
    dataset_train = AbstractDataset(encoded_train, label_train)
    dataset_val = AbstractDataset(encoded_val, label_val)
    # dataset_test = AbstractDataset(encoded_test, label_test)

    # time.strftime("%Y%m%d-%H%M%S")
    ending = str(args.maxlength) + '_' + str(args.epochs) + '_' + str(args.tbatch) + '_' + str(args.lr) + '_' + str(args.weightdecay)
    path = os.path.join(path, ending)
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path + '/results'):
        os.mkdir(path + '/results')
    if not os.path.exists(path + '/logs'):
        os.mkdir(path + '/logs')

    #os.mkdir(path)
    #os.mkdir(path + '/results')
    #os.mkdir(path + '/logs')

    training_args = transformers.TrainingArguments(
        learning_rate = args.lr,
        num_train_epochs=args.epochs,              
        per_device_train_batch_size=args.tbatch,  
        per_device_eval_batch_size=args.vbatch,   
        warmup_steps=args.warmup,                
        weight_decay=args.weightdecay,               
        output_dir=path + '/results',      
        overwrite_output_dir=args.overwrite,
        save_total_limit=args.savelim,
        evaluation_strategy='steps',
        save_strategy='steps',
        load_best_model_at_end=args.loadbest,    
        logging_dir=path + '/logs',            
        logging_steps=args.loggingsteps
    )

    trainer = transformers.Trainer(
        model=model,                         
        args=training_args,
        compute_metrics=compute_metrics,                  
        train_dataset=dataset_train,         
        eval_dataset=dataset_val             
    )

    trainer.train(resume_from_checkpoint=args.loadfromfile)

    save_stats(trainer, path)

def parsearg_utils():
    import argparse

    parser = argparse.ArgumentParser(description='Run Hugging Face model from transformer library on abstract data.')

    parser.add_argument('-p','--path', help='Path to save data model and data, if applicable (str)', default='/content/drive/My Drive/nlu_project/', type=str)
    parser.add_argument('-d','--data', help='Path to dataframe to load (str)', default='/content/drive/My Drive/nlu_project/assets/articles.csv', type=str)
    parser.add_argument('-m','--model', help='Model name: SciBERT:allenai/scibert_scivocab_uncased (default), BioBERT:dmis-lab/biobert-v1.1, BioMed-RoBERTa:allenai/biomed_roberta_base, BlueBERT:bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12 (str)', default='SciBERT:allenai/scibert_scivocab_uncased', type=str)
    parser.add_argument('-n','--maxlength', help='Maximum length (int)', default=256, type=int)
    parser.add_argument('-l','--lr', help='Learning rate (float)', default=0.00001, type=float)
    parser.add_argument('-e','--epochs', help='Number of training epochs (int)', default=3, type=int)
    parser.add_argument('-t','--tbatch', help='Training batch size (int)', default=16, type=int)
    parser.add_argument('-v','--vbatch', help='Validation batch size (int)', default=16, type=int)
    parser.add_argument('-w','--warmup', help='Number of warm-up steps (int)', default=500, type=int)
    parser.add_argument('-c','--weightdecay', help='Weight decay (float)', default=0.1, type=float)
    parser.add_argument('-o','--overwrite', help='Overwrite output directory (bool)', default=True, type=bool)
    parser.add_argument('-s','--savelim', help='Save total limit (int)', default=2, type=int)
    parser.add_argument('-b','--loadbest', help='Load best model at end (bool)', default=True, type=bool)
    parser.add_argument('-g','--loggingsteps', help='Logging steps (int)', default=100, type=int)
    parser.add_argument('-f','--loadfromfile', help='Load model from checkpoint (bool)', default=False, type=bool)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    arguments = parsearg_utils()
    main(arguments)