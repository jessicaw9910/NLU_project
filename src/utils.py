#!/usr/bin/env python3

import numpy as np
import pandas as pd
import time
import random
from pymed import PubMed
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def save_abstracts(df_input, drug=None, tcga=None, abstract_list=None,
                   tool="PubMedSearcher", email="jwh4001@med.cornell.edu", first_pass=False):
    ## only run the first time
    if first_pass:
        abstract_list = []
        DRUGS = df_input['DRUG_NAME']
        INDICATIONS = df_input['study_name']
 
    ## if error pick up where the for loop left off
    else:
        idx = df_input.index[(df_input['DRUG_NAME'] == drug) & (df_input['study_name'] == tcga)]
        n = len(df_input['DRUG_NAME'])

        DRUGS = df_input['DRUG_NAME'][int(idx[0]):n]
        INDICATIONS = df_input['study_name'][int(idx[0]):n]

    try:
        for drug, tcga in zip(DRUGS, INDICATIONS):
        
            search_term = drug + " " + tcga
            #start = time.time()
            results = pubmed.query(search_term, max_results=500)
            #end = time.time()
            articleList = []

            ## not to trigger 429 error
            ## 3 requests max per second
            #if end - start < 0.3:
            time.sleep(0.5)

            for article in results:
            ## convert PubMedArticle to dictionary
                articleDict = article.toDict()
                articleList.append(articleDict)

            ## list of dict records to hold article details fetched from PUBMED API
            for article in articleList:
                ## sometimes article['pubmed_id'] contains list separated with comma
                ## take first pubmedId in that list - thats article pubmedId
                pubmedId = article['pubmed_id'].partition('\n')[0]
                title = article['title']
                abstract = article['abstract']
                
                ## skip if abstract is None, '', or < 100 tokens
                if abstract is None or abstract == '' or len(abstract.split(' ')) < 100:
                    continue

                ## skip if title None
                if article['abstract'] is None:
                    continue

                ## skip None type before converting to UTF-8
                abstract = abstract.replace('\n', ' ').encode('ascii',errors='ignore').decode("utf-8")

                ## skip if drug name not in abstract
                if abstract.lower().find(drug.lower()) == -1:
                    continue

                ## append article info to dictionary 
                abstract_list.append({u'drug': drug,
                                      u'indication': tcga,
                                      u'pubmed_id': pubmedId,
                                      u'title': title,
                                      #u'journal':article['journal'],
                                      u'abstract': abstract})

        return abstract_list, drug, tcga

    except:
        return abstract_list, drug, tcga

def create_dataframe(abstract_dict, df_drug):
    ## create dataframe for abstracts
    df_abstract = pd.DataFrame.from_dict(abstract_dict)
    # sum([len(i.split(' ')) > 512 for i in df_abstract['abstract']])
    ## unique ID for each drug + indication pair
    df_abstract['id'] = df_abstract['drug'] + '_' + df_abstract['indication']

    ## add ID to df_mad and column for whether or not abstract exists
    df_drug['id'] = df_drug['DRUG_NAME'] + '_' + df_drug['study_name']
    ids = list(np.unique(df_abstract['id']))
    df_drug['abstract_bool'] = list(map(lambda x: x in ids, df_drug['id']))

    ## add sensitivity info to articles df
    df_abstract = pd.merge(df_abstract, df_drug[['SENSITIVE', 'id']], on=['id'], how='inner')

    return df_abstract

def run_baseline(df_drug, seed=123):
    n_true = sum(df_drug[(df_drug['abstract_bool'] == True)]['SENSITIVE'])
    n_false = len(df_drug[(df_drug['abstract_bool'] == True)]['SENSITIVE']) - n_true

    list_random = [1] * n_true + [0] * n_false

    y = df_drug[(df_drug['abstract_bool'] == True)]['SENSITIVE']
    n = len(y)

    random.seed(seed)
    list_acc = []
    list_precision = []
    list_recall = []

    for idx in range(1000):
        random.shuffle(list_random)
        list_acc.append(sum(list_random != y) / n)

        yhat_y1 = [list_random[i] for i in np.where(y == 1)[0]]
        yhat_y0 = [list_random[i] for i in np.where(y == 0)[0]]

        tp = sum(yhat_y1)
        fp = len(yhat_y1) - tp
        fn = sum(yhat_y0)
        # tn = len(yhat_y0) - fn

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        list_precision.append(precision)
        list_recall.append(recall)

    list_f1 = [2 * (p * r) / (p + r) for p, r in zip(list_precision, list_recall)]

    return list_acc, list_f1

def annotate_splits(df_abstract, pct_val=0.15, pct_test=0.15, seed=123):
    ## find and shuffle unique IDs
    id = df_abstract['id'].unique()
    random.seed(seed)
    random.shuffle(id)

    ## create arrays of train, val, and test IDs
    n = len(id)
    idx_train = round(n * (1-pct_val) * (1-pct_test))
    idx_val = round(n * pct_val * (1-pct_test))
    id_train = id[0:idx_train]
    id_val = id[idx_train:(idx_train + idx_val)]
    id_test = id[(idx_train + idx_val):len(id)]
    # len(id_train) + len(id_val) + len(id_test)

    ## create training, validation, and test dataframes
    # df_train = df_abstract[df_abstract['id'].isin(id_train)]
    # df_val = df_abstract[df_abstract['id'].isin(id_val)]
    # df_test = df_abstract[df_abstract['id'].isin(id_test)]
    # df_val.shape[0] / df_abstract.shape[0]

    ## add training, validation, and test label to abstracts df
    df_abstract['set'] = np.nan
    df_abstract.loc[df_abstract['id'].isin(id_train), 'set'] = 'train'
    df_abstract.loc[df_abstract['id'].isin(id_val), 'set'] = 'validation'
    df_abstract.loc[df_abstract['id'].isin(id_test), 'set'] = 'test'
    # df_abstract['set'].value_counts()

    return df_abstract

class AbstractDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def save_stats(trainer, path):
  n = len(trainer.state.log_history)

  train_list = []
  eval_list = []

  for idx, obj in enumerate(trainer.state.log_history):
    if idx % 2 == 0 and (idx + 1) < n:
      train_list.append(obj)
    if idx % 2 != 0 and (idx + 1) < n:
      eval_list.append(obj)
    if idx + 1 == n:
      summary_stats = obj

  df_train = pd.DataFrame.from_dict(train_list)
  df_train.to_csv (path + '/df_train.csv', index = False, header=True)

  df_eval = pd.DataFrame.from_dict(eval_list)
  df_eval.to_csv (path + '/df_eval.csv', index = False, header=True)

  df_summary = pd.DataFrame(summary_stats, index=[0])
  df_summary.to_csv (path + '/df_summary.csv', index = False, header=True)