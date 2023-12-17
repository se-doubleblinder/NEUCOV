import os
import pickle
import random
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer, RobertaConfig

from dataset import CodeCoveragePredictionDataset
from model import CodeCoveragePredictionModel

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_metrics(label_pairs):
    # Statement Match
    true = label_pairs['true']
    pred = label_pairs['preds']
    
    flattened_true = [item for sublist in true for item in sublist]
    flattened_pred = [item for sublist in pred for item in sublist]
    
    assert len(flattened_true) == len(flattened_pred)
    num_matches = sum(1 for i, x in enumerate(flattened_pred) if x == flattened_true[i])
    accuracy_percentage = (num_matches / len(flattened_true)) * 100.0
    em_accuracy_micro = accuracy_percentage
    
    # Exact Match
    exact_match_scores = []
    for true_sublist, pred_sublist in zip(true, pred):
        if true_sublist == pred_sublist:
            exact_match_scores.append(100)
        else:
            exact_match_scores.append(0)
    em_accuracy = sum(exact_match_scores)/len(exact_match_scores)   
    
    metrics = {
        'EM-Accuracy': em_accuracy,
        'Statement Match' : em_accuracy_micro,
        'Accuracy': accuracy_score(flattened_true, flattened_pred),
        'Precision': precision_score(flattened_true, flattened_pred),
        'Recall': recall_score(flattened_true, flattened_pred),
        'F1-Score': f1_score(flattened_true, flattened_pred),
    }
    return metrics


def evaluate(eval_dataloader, model, args, epoch_stats=None):
    total_eval_loss = 0
    label_pairs = {'true': [], 'preds': []}

    logger.warning("***** Running evaluation *****")
    logger.warning(f"  Num examples = {len(eval_dataloader)}")
    logger.warning(f"  Batch size = {args.eval_batch_size}")

    if not epoch_stats:
        epoch_stats = {}

    for batch in tqdm(eval_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        # Tell pytorch not to bother with constructing the compute graph during the
        # forward pass, since this is only needed for backpropogation (training).
        with torch.no_grad():
            batch_loss, batch_preds, batch_true = model(batch[0], batch[1], batch[2], batch[3], batch[4])
            # Accumulate the validation loss.
            total_eval_loss += batch_loss.item()

        # Move labels to CPU
        curr_true = [x.tolist() for x in batch_true]
        label_pairs['true'] += curr_true

        curr_preds = []
        for item_preds in batch_preds:
            curr_preds.append([1 if x > 0.5 else 0 for x in item_preds.tolist()])
        label_pairs['preds'] += curr_preds

    # Calculate the average loss over all of the batches.
    eval_loss = total_eval_loss / len(eval_dataloader)
    eval_metrics = compute_metrics(label_pairs)
 
    # Record all statistics.
    return {
        **epoch_stats,
        **{'Epoch evaluation loss': eval_loss},
        **eval_metrics,
    }, label_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='./Dataset/training.json', type=str,
                        help="Path to datasets directory.")
    parser.add_argument("--model_key", default='microsoft/codeexecutor',
                        type=str, help="Model string.",
                        choices=['microsoft/codeexecutor', 'microsoft/graphcodebert-base', 'microsoft/codebert-base', 'roberta-base'])
    parser.add_argument("--output_dir", default='./Dataset/outputs/',
                        type=str, help="Dataset type string.")

    ## Model parameters
    parser.add_argument("--max_tokens", default=512, type=int,
                        help="Maximum number of tokens in a statement")
    parser.add_argument("--use_statement_ids", action='store_true',
                        help="Use statement ids in input embeddings")

    ## Experiment arguments
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to predict on given dataset.")
    parser.add_argument("--pretrain", action='store_true',
                        help='Use xBERT model off-the-shelf')
    parser.add_argument("--save_predictions", action='store_true',
                        help='Cache model predictions during evaluation.')
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay for Adam optimizer.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--max_source_size", default=512, type=int,
                        help="Optional input sequence length after tokenization.")

    # Print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()

    logger.warning(f"Device: {args.device}, Number of GPU's: {args.n_gpu}")

    # Set seed
    set_seed(args.seed)

    # Make directory if output_dir does not exist
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        Path(output_dir).mkdir(exist_ok=True, parents=True)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_key)
    special_tokens_list = ['<line>', '<state>', '</state>', '<dictsep>', '<output>', '<indent>',
                            '<dedent>', '<mask0>']
    for i in range(len(special_tokens_list)):
        special_tokens_list.append(f"<{i}>")
    special_tokens_dict = {'additional_special_tokens': special_tokens_list}
    tokenizer.add_special_tokens(special_tokens_dict)

    config = RobertaConfig.from_pretrained(args.model_key)
    model = CodeCoveragePredictionModel(args, config, tokenizer)

    if args.load_model_path is not None:
        logger.info(f"Reload model from {args.load_model_path}")
        model.load_state_dict(torch.load(args.load_model_path), strict=False)

    model.to(args.device)
    print(model)
    print()
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
    print(args)

    if args.do_train:
        logger.info('Loading training data.')
        train_dataset = CodeCoveragePredictionDataset(tokenizer, args.data_dir, args, logger)
        logger.info('Constructing dataloader for training data.')
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.train_batch_size, drop_last=False)

        logger.info('Loading validation data.')
        val_dataset = CodeCoveragePredictionDataset(tokenizer, './Dataset/validation.json', args, logger)
        logger.info('Constructing dataloader for Validation data.')
        val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), 
                                    batch_size=args.eval_batch_size, drop_last=False)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                          eps=args.adam_epsilon)
        max_steps = len(train_dataloader) * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps*0.1,
                                                    num_training_steps=max_steps)
        # Start training
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Batch size = {args.train_batch_size}")
        logger.info(f"  Num epochs = {args.num_train_epochs}")

        training_stats = []
        model.zero_grad()

        for epoch in range(args.num_train_epochs):
            training_loss, num_train_steps = 0, 0
            label_pairs = {
                'true': [], 'preds': []
            }

            model.train()
            for batch in tqdm(train_dataloader):                
                batch = tuple(t.to(args.device) for t in batch)
                batch_loss, batch_preds, batch_true = model(*batch)
                training_loss += batch_loss.item()
                curr_true = [x.tolist() for x in batch_true]
                label_pairs['true'] += curr_true

                curr_preds = []
                for item_preds in batch_preds:
                    curr_preds.append([1 if x > 0.5 else 0 for x in item_preds.tolist()])
                label_pairs['preds'] += curr_preds
                num_train_steps += 1

                batch_loss.backward()
                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                # Update the learning rate
                scheduler.step()

            epoch_tr_loss = training_loss / len(train_dataloader)
            epoch_eval_metrics = compute_metrics(label_pairs)
            epoch_acc = epoch_eval_metrics['Accuracy']

            logger.info(f"Epoch {epoch}, Training loss: {epoch_tr_loss}")
            logger.info(f"Epoch {epoch}, Training accuracy for Code Coverage prediction: {epoch_acc}")
            
            # After the completion of one training epoch, measure performance
            # on validation set.
            logger.info('Measuring performance on validation set.')
            # Put the model in evaluation mode--the dropout layers behave
            # differently during evaluation.
            model.eval()
            training_stats, label_pairs = evaluate(val_dataloader, model, args,
                                         epoch_stats={
                                            'Epoch training loss': epoch_tr_loss,
                                            'Epoch accuracy': epoch_acc,
                                         }
                                )
            print(training_stats)

            epoch_output_dir = Path(output_dir) / f'Epoch_{epoch}'
            epoch_output_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Saving model to {epoch_output_dir}")
            torch.save(model.state_dict(), str(epoch_output_dir / 'model.ckpt'))
                    
    if args.do_eval:
        # Put the model in evaluation mode--the dropout layers behave
        # differently during evaluation.
        model.eval()
        logger.info('Loading test data...')
        test_dataset = CodeCoveragePredictionDataset(tokenizer, args.data_dir, args, logger)
        test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.eval_batch_size, drop_last=False)
        logger.info('Constructing data loader for testing data.')

        stats, label_pairs  = evaluate(test_dataloader, model, args)
        print("  EM-Accuracy: {0:.4f}".format(stats['EM-Accuracy']))
        print("  Statement Match: {0:.4f}".format(stats['EM-Accuracy']))
        print("  Test loss: {0:.4f}".format(stats['Epoch evaluation loss']))
        print("  Accuracy: {0:.4f}".format(stats['Accuracy']))
        print("  Precision: {0:.4f}".format(stats['Precision']))
        print("  Recall: {0:.4f}".format(stats['Recall']))
        print("  F1-Score: {0:.4f}".format(stats['F1-Score']))
        
        if args.save_predictions:
            with open(str(output_dir / 'predictions.pkl'), 'wb') as f:
                pickle.dump(label_pairs, f)