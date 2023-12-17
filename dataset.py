import json
import re
import torch
import ast
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
        
class InputFeatures(object):
    def __init__(self, code_tokens, sentence_id, test_variable_lines, coverage_labels, id):
        self.code_tokens = code_tokens
        self.sentence_id = sentence_id
        self.test_variable_lines = test_variable_lines
        self.coverage_labels = coverage_labels
        self.id = id
        
def extract_variable(code):
    line_numbers = []  
    variables = [] 
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.If, ast.While, ast.FunctionDef)):
            break
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    variables.append(target.id)
                    line_numbers.append(target.lineno - 1)
    return line_numbers
    
def indent_dedent_tokenize(code, tokenizer):
    current_num_white_spaces = 0
    prev_num_white_spaces = 0
    code_tokens_list = []
    num_tokens = 0
    sentence_id = []
    
    lines = code.split('\n')
    for line_number, line in enumerate(lines):
        indented_code = ""
        line_number_token = f"<{line_number}>"
        indented_code += line_number_token + " "
        current_num_white_spaces = len(re.match(r'^\s*', line).group(0))
        
        if current_num_white_spaces > prev_num_white_spaces:
            indented_code += "<indent> "
            prev_num_white_spaces = current_num_white_spaces
        elif current_num_white_spaces < prev_num_white_spaces:
            diff = prev_num_white_spaces - current_num_white_spaces
            for temp in range(diff // 4):
                indented_code += "<dedent> "
            prev_num_white_spaces = current_num_white_spaces
            
        line = line.lstrip()
        indented_code += line + "\n"     
        code_tokens_of_one_line = tokenizer.tokenize(indented_code)
        num_tokens += len(code_tokens_of_one_line)
        sentence_id.extend([line_number] * len(code_tokens_of_one_line))
        
        for token in code_tokens_of_one_line:
            code_tokens_list.append(token)
            
    return code_tokens_list , sentence_id

def convert_examples_to_features(js, tokenizer):
    id = js["id"]
    code_tokens , sentence_id = indent_dedent_tokenize(js["code"], tokenizer)
    test_variable_lines = extract_variable(js["code"])
    return InputFeatures(code_tokens, sentence_id, test_variable_lines, js["coverage"], id)
    
class CodeCoveragePredictionDataset(Dataset):
    def __init__(self, tokenizer, dataPath, args,logger):
        self.args = args
        self.tokenizer = tokenizer
        self.examples = []

        with open(dataPath, "r") as json_file:
            dataset = json.load(json_file)
        print(len(dataset))
        error_num = 0
        num_example = 0
        for index in dataset:
            if len(dataset[index]['code']) != 0:
                coverage_labels = [1 if element == '>' else 0 for element in dataset[index]["coverage"]]
                has_zero = any(label == 0 for label in coverage_labels)
                has_one = any(label == 1 for label in coverage_labels)
                # if has_zero and has_one: 
                js = {
                    "id": dataset[index]['id'],
                    "code": dataset[index]["code"],
                    "coverage": coverage_labels,
                }
                try:
                    features = convert_examples_to_features(js, tokenizer)
                    self.examples.append(features)
                    num_example += 1
                except Exception as e:
                    error_num += 1
                    print(f"Error processing example: {str(e)}")

        logger.warning(f"Num examples = {len(self.examples)}")
        logger.warning(f"Error num = {error_num}")

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item): 
        js = self.examples[item]
        max_source_size = self.args.max_source_size

        # Encoder-Decoder for Trace Generation
        source_tokens = js.code_tokens[ : max_source_size ] 
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)

        source_masks = [1 for _ in range(len(source_ids))]
        zero_padding_length = max_source_size - len(source_ids)
        
        source_ids += [self.tokenizer.pad_token_id for _ in range(zero_padding_length)]
        source_masks += [self.tokenizer.pad_token_id for _ in range(zero_padding_length)]
        
        
        gold_padding_length = max_source_size - len(js.coverage_labels)
        gold_ids = js.coverage_labels + [-999] * gold_padding_length
        
        sentence_id = js.sentence_id[ : max_source_size] 
        sentence_id_padding_length = max_source_size - len(sentence_id)
        sentence_id += [-999] * sentence_id_padding_length
        js.sentence_id = sentence_id
        
        test_variable_lines_padding_length = max_source_size - len(js.test_variable_lines)
        js.test_variable_lines += [-999] * test_variable_lines_padding_length

        return (
            torch.tensor(source_ids),
            torch.tensor(source_masks),
            torch.tensor(js.sentence_id),
            torch.tensor(js.test_variable_lines),
            torch.tensor(gold_ids),
        )
