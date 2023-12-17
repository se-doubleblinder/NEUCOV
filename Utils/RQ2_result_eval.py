import ast
import json

# Lists to store the results
true_counter = 0
false_counter = 0
total_counter = 0
counterr = 0
# Function to find all method names in the code
def find_method_definitions(node):
    method_definitions = {}
    for subnode in ast.walk(node):
        if isinstance(subnode, ast.FunctionDef):
            method_name = subnode.name
            start_line = subnode.lineno
            end_line = subnode.end_lineno
            method_definitions[method_name] = (start_line, end_line)
    return method_definitions

# Function to find the line numbers from which methods are being called
def find_method_calls(node, method_name):
    calls = []
    for subnode in ast.walk(node):
        if isinstance(subnode, ast.Call):
            if isinstance(subnode.func, ast.Name) and subnode.func.id == method_name:
                calls.append(subnode.lineno)
    return calls

def get_final_pred(unique_preds, ground_truth):
    # Ground Truth length
    gt_len = len(ground_truth)
    pred = [0] * gt_len
    for i in range(len(unique_preds)):        
        line_number = unique_preds[i]
        if line_number < len(pred):
            pred[line_number] = 1
    return pred

def operation(instance_ranges, preds, ground_truth, method_call_line):
    global true_counter, false_counter, total_counter, counterr

    for instance_range in instance_ranges:
        flag = False
        start , end = instance_range
        start = start - 1
        end = end - 1
        
        if ground_truth[method_call_line - 1] == 1:
            counterr += 1
            if preds[method_call_line - 1] == 1:
                for k in range(start, end + 1):
                    if preds[k] == 1:
                        flag = True
                        break
                if flag:
                    true_counter += 1
                    total_counter += 1
                else:
                    false_counter += 1
                    total_counter += 1
            else:
                continue
       
        elif ground_truth[method_call_line - 1] == 0:
            counterr += 1
            if preds[method_call_line - 1] == 0:
                for k in range(start, end + 1):
                    if preds[k] == 1:
                        flag = True
                        break
                if flag:
                    false_counter += 1
                    total_counter += 1
                else:
                    true_counter += 1
                    total_counter += 1 
            else:
                continue      

if __name__ == "__main__":
    with open('../Dataset/multiple_method.json', 'r') as json_file:
        data = json.load(json_file)
    with open('LOAD PREDICTION FILE HERE', 'r') as json_file:
        output = json.load(json_file)

    # Lists to store the results
    method_call_line_numbers = []
    method_definition_ranges = []

    for j, index in enumerate(data):
        if index == '12446':
            continue
        python_code = data[index]['code']
        total_lines = len(python_code.split('\n'))

        final_pred = output['preds'][j]
        ground_truth = output['true'][j]

        # pred_coverage = output[j]['preds_topK']
        # unique_preds = list(set(pred_coverage))

        # grd_truth = output[j]['gold']
        # ground_truth = [int(x) for x in grd_truth]
        # final_pred = get_final_pred(unique_preds, ground_truth)

        assert total_lines == len(final_pred) == len(ground_truth)

        # Parse the Python code into an abstract syntax tree (AST)
        try:
            parsed_code = ast.parse(python_code)
        except Exception as e:
            print(e)

        try:
            method_definitions = find_method_definitions(parsed_code)
        except Exception as e:
            print(e)
        
        instance_method_call_line_numbers = []
        instance_method_definition_ranges = []

        for method_name, (start_line, end_line) in method_definitions.items():
            calls = find_method_calls(parsed_code, method_name)
            instance_method_call_line_numbers.extend(calls)
            instance_method_definition_ranges.append((start_line, end_line))

        # Append lists for this instance to the overall results
        method_call_line_numbers.append(instance_method_call_line_numbers)
        method_definition_ranges.append(instance_method_definition_ranges)

        # Append lists for this instance to the overall results
        for method_call_line in instance_method_call_line_numbers:
            operation(instance_method_definition_ranges, final_pred, ground_truth, method_call_line)
    print("True: ", true_counter)
    print("False: ", false_counter)
    print("Total: ", total_counter)
    print("Accuracy: ", true_counter / total_counter)
    print("Counterr: ", counterr)   