import json
import ast
import ast
from pprint import pprint

# AST Visitor
class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.stats = {'if': [], 'for': [], 'while': []}
        self.current_blocks = []
        self.filtered_blocks = []

    def visit_If(self, node):
        self.current_blocks.append({
            'type': 'if',
            'start_line': node.lineno,
            'end_line': node.end_lineno,
        })
        self.generic_visit(node)
        self.current_blocks.pop()

    def visit_For(self, node):
        self.current_blocks.append({
            'type': 'for',
            'start_line': node.lineno,
            'end_line': node.end_lineno,
        })
        self.generic_visit(node)
        self.current_blocks.pop()

    def visit_While(self, node):
        self.current_blocks.append({
            'type': 'while',
            'start_line': node.lineno,
            'end_line': node.end_lineno,
        })
        self.generic_visit(node)
        self.current_blocks.pop()

    def check_subset(self, block):
        for existing_block in self.filtered_blocks:
            if (
                block['start_line'] >= existing_block['start_line']
                and block['end_line'] <= existing_block['end_line']
            ):
                return True
        return False

    def filter_blocks(self):
        for block in self.current_blocks:
            if not self.check_subset(block):
                self.filtered_blocks.append(block)

    def visit(self, node):
        self.filter_blocks()
        super().visit(node)

    def report(self):
        self.filter_blocks()
        pprint(self.filtered_blocks)

    def get_line_ranges(self):
        line_ranges = []
        for block in self.filtered_blocks:
            line_ranges.append((block['start_line'] - 1, block['end_line'] - 1))
        return line_ranges
    
# Get  Branches
def get_branches_line(code):
    tree = ast.parse(code)
    a = Analyzer()
    a.visit(tree)
    # a.report()
    line_ranges = a.get_line_ranges()
    return line_ranges

# Preprocess the prediction
def get_final_pred(unique_preds, ground_truth):
    # Ground Truth length
    gt_len = len(ground_truth)
    pred = [0] * gt_len
    for i in range(len(unique_preds)):        
        line_number = unique_preds[i]
        if line_number < len(pred):
            pred[line_number] = 1
    return pred

# Get Coverage List
def get_coverage_list(code, branch_lines, pred_coverage, gold_coverage, dataset_coverage):
    coverage_within_branch = []
    for b_range in branch_lines:
        start_line = b_range[0]
        end_line = b_range[1]
        temp_dict = {}
        temp_pred = []
        temp_gold = []
        if len(pred_coverage) == len(dataset_coverage) and len(gold_coverage) == len(dataset_coverage):
            for i in range(start_line, end_line + 1):
                if i < len(pred_coverage):
                    temp_pred.append(pred_coverage[i])
                    temp_gold.append(gold_coverage[i])
            
            temp_dict['pred'] = temp_pred
            temp_dict['gold'] = temp_gold
            coverage_within_branch.append(temp_dict)

    return coverage_within_branch

# Accuracy
def macro_accracy(list1, list2):
    assert len(list1) == len(list2)
    num_matches = sum(1 for a, b in zip(list1, list2) if a == b)
    accuracy = (num_matches / len(list1)) * 100.0
    return accuracy

def micro_accracy(f_pred,f_gold):
    assert len(f_pred) == len(f_gold)
    # Check how many elements in list1 match with list2
    num_matches = sum(1 for i, x in enumerate(f_pred) if x == f_gold[i])
    # Calculate the accuracy percentage
    accuracy_percentage = (num_matches / len(f_gold)) * 100.0
    return accuracy_percentage
        
# Starting Point of the Code
with open('JSON FILE PATH') as f:
    data = json.load(f)

with open('JSON FILE PATH') as f:
    data1 = json.load(f)

f_pred = []
f_gold = []
macro_accuracy_array = []

for index, i in enumerate(data):
    code = data[i]['code']
    # Dataset Coverage
    dataset_coverage = data[i]['coverage']

    is_neuralCC_coverage_Eval = False
    is_neuralCC_codenetMut = True 
    is_codeExe_coverageEval = False
    is_codeExe_codenetMut = False

    if is_neuralCC_coverage_Eval:
        gold_coverage = data1['true'][index]
        pred_coverage = data1['preds'][index]
    elif is_neuralCC_codenetMut:
        gold_coverage = data1['true'][index]
        pred_coverage = data1['preds'][index]

    # branch Locator
    branch_lines = get_branches_line(code)
    # Get the Coverage List from the Branch Lines
    coverage_from_branch = get_coverage_list(code, branch_lines, pred_coverage, gold_coverage, dataset_coverage)
    # Evaluate the Coverage
    for c in coverage_from_branch:
        pred = c['pred']
        gold = c['gold']
        f_gold += gold
        f_pred += pred
        macro_accuracy = macro_accracy(pred, gold)
        macro_accuracy_array.append(macro_accuracy)
micro_statement_match_value = micro_accracy(f_pred, f_gold)

print(f'Micro Accuracy: {micro_statement_match_value}')
print(f'Macro Accuracy: {sum(macro_accuracy_array)/len(macro_accuracy_array)}')