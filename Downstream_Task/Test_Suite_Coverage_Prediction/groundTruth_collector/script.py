import json
from test_function import test_function
import subprocess
import coverage

with open('../dataset/final_Incomplete_Code_Dataset.json', 'r') as file:
    data = json.load(file)

def write_code_to_test_function(submission_code):
    with open('test_function.py', 'w') as file:
        file.write("def test_function():\n")
        lines = submission_code.split('\n')
        line_index = 0
        main_block_found = False
        method_call = "main()"
        while line_index < len(lines):
            current_line = lines[line_index]
            if 'if __name__' in current_line and '__main__' in current_line:
                main_block_found = True
                main_line_split = current_line.split(':')
                if len(main_line_split) > 1 and main_line_split[1].strip():
                    method_call = main_line_split[1].strip()
                    line_index += 1
                    continue
                elif line_index + 1 < len(lines) and lines[line_index + 1].strip():
                    method_call = lines[line_index + 1].strip()
                    line_index += 2
                    continue
            file.write("    " + current_line + "\n")
            line_index += 1
        if main_block_found:
            file.write("    " + method_call + "\n")        
        file.write("test_function()")

def calculate_lines():
    with open('test_function.py', 'r') as file:
        lines = file.readlines()
    return len(lines)

def parse_missing_lines(line_string):
    parts = line_string.split(',')
    lines = []
    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            lines.extend(range(start, end + 1))
        else:
            lines.append(int(part))

    return lines

def create_execution_array(total_lines, non_executed_lines):
    execution_array = [1] * total_lines

    # Indexing starts from 0
    for i in range(len(non_executed_lines)):
        non_executed_lines[i] = non_executed_lines[i] - 2
    
    for i in non_executed_lines:
        execution_array[i] = 0
            
    return execution_array


ground_truth_data = {}
final_error_dict = {}

# Starting point of the code
for index, id in enumerate(data):
    testCase_data = {}
    error_dict = {}
    for tCode in data[id]:

        code = data[id][tCode]
        
        write_code_to_test_function(code)

        lines_of_code = calculate_lines() - 2

        subprocess.run(['coverage', 'run', '--source=.', 'test_function.py'])
        subprocess.run(['coverage', 'html'])

        cov = coverage.Coverage()
        cov.load()
        filename = 'test_function.py'
        file_analysis = cov.analysis(filename)
        executed_lines = file_analysis[1]
        missing_lines = file_analysis[3]
        try:
            missing_lines = parse_missing_lines(missing_lines)
            execution_array = create_execution_array(lines_of_code, missing_lines)
            testCase_data[tCode] = {'code': code, 'execution_array': execution_array}
        except:
            error_dict[tCode] = {'code': code, 'missing_lines': missing_lines}
            continue

    ground_truth_data[id] = testCase_data

    if (index + 1) % 5 == 0:
        with open(f'../dataset/completeCode_ground_truth{index + 1}.json', 'w') as file:
            json.dump(ground_truth_data, file, indent=4)
        ground_truth_data = {}