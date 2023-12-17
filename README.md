# Predictive Code Coverage Analysis
Code coverage is a key measure of testing effectiveness, which evaluates the extent to which a test suite exercises various aspects of the code, including statements, branches, or paths. However, coverage profilers require access to the entire codebase, limiting their applicability in scenarios where the code is incomplete, or execution is not desirable and even prohibitively expensive. To address these challenges, this paper introduces NeuralCC, a novel approach that leverages pre-trained language models (PLMs) that possess an understanding of the intricacies of code execution. Our tool takes advantage of the code execution-specific pre-training to learn the dynamic program dependencies, including both dynamic data and control dependencies between the input and statements, leading to an improved code coverage computation. 

We conducted several experiments to evaluate NeuralCC. The results showed that NeuralCC achieves high accuracy up to 73.11% in exact-match, 94.26% in statement-match, and 87.87% in branch-match, and performed relatively better than the baselines up to1.49x, 43.7%, and 60.53% in those metrics. Our in-depth analysis revealed that NeuralCC is able to learn the inter-procedural control flows across multiple methods with 85.26% accuracy, enabling accurate inter-procedural code coverage analysis. We also showed NeuralCC’s usefulness in the downstream task of predicting the least covered statements with high accuracy.

## Getting Started with NeuralCC


### Pre-Trained Model/Tokenizer Asset Links

- Here is the link for NeuralCC model:
  - [Model](https://drive.google.com/file/d/10eyuGINrsiDE_P2kiXY9qYK47caMuuBf/view?usp=sharing)


### Run Instructions
  
```
$ python run.py --help     
usage: run.py [-h] [--data_dir DATA_DIR] [--model_key {microsoft/codebert-base,microsoft/graphcodebert-base,roberta-base}] [--output_dir OUTPUT_DIR] [--max_tokens MAX_TOKENS] [--pretrain] [--use_statement_ids][--load_model_path LOAD_MODEL_PATH] [--do_train] [--do_eval] [--save_predictions] [--train_batch_size TRAIN_BATCH_SIZE] [--eval_batch_size EVAL_BATCH_SIZE] [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--adam_epsilon ADAM_EPSILON]
[--num_train_epochs NUM_TRAIN_EPOCHS] [--seed SEED]

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Path to datasets directory.
  --model_key {microsoft/codebert-base,microsoft/graphcodebert-base,roberta-base}
                        Model string.
  --output_dir OUTPUT_DIR
                        Dataset type string.
  --max_tokens MAX_TOKENS
                        Maximum number of tokens in a statement

  --load_model_path LOAD_MODEL_PATH
                        Path to trained model: Should contain the .bin files
  --do_train            Whether to run training.
  --do_eval             Whether to run eval on the dev set.

  --pretrain            Use xBERT model off-the-shelf
  --use_statement_ids   Use statement ids in input embeddings 

  --save_predictions    Cache model predictions during evaluation.
  --train_batch_size TRAIN_BATCH_SIZE
                        Batch size per GPU/CPU for training.
  --eval_batch_size EVAL_BATCH_SIZE
                        Batch size per GPU/CPU for evaluation.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --weight_decay WEIGHT_DECAY
                        Weight decay for Adam optimizer.
  --adam_epsilon ADAM_EPSILON
                        Epsilon for Adam optimizer.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --seed SEED           random seed for initialization

``` 

### Sample Commands for Replicating Experiments:
1. Training
```bash
$ python run.py --data_dir  ./Dataset/training.json --output_dir ./Dataset/outputs/ --do_train
```
2. Inference
```bash
$ python run.py --data_dir ./Dataset/testing.json  --output_dir ./Dataset/outputs/ --do_eval --load_model_path ./Dataset/outputs/model.ckpt
```

### Dependencies:
```bash
  $ pip install -r requirements.txt
```