import json
import random
import re

# Dataset File

# Complete Code
# with open('./dataset/raw_code_dataset.json', 'r') as file:
#     data = json.load(file)

# incomplete Code
with open('./dataset/raw_incomplete_code.json', 'r') as file:
    data = json.load(file)

# Test Cases Files
with open('./testCases/integers/int_testcases.json', 'r') as file:
    int_testcases = json.load(file)
with open('./testCases/list/list_int_testcases.json', 'r') as file:
    list_int_testcases = json.load(file)
with open('./testCases/strings/string_of_int_testcases.json', 'r') as file:
    string_of_int_testcases = json.load(file)
with open('./testCases/strings/string_testcases.json', 'r') as file:
    string_testcases = json.load(file)

print(len(data))
print(len(int_testcases))
print(len(list_int_testcases))
print(len(string_of_int_testcases))
print(len(string_testcases))

lineCoverage_dict = {}

for id in data:
    given_dtype = ''
    test_case_list = []
    code = data[id]['code']
    data_type = data[id]['dType']
    if data_type == 'int': given_dtype = int_testcases
    if data_type == 'list_int': given_dtype = list_int_testcases
    if data_type == 'string': given_dtype = string_testcases
    test_case_list = random.sample(given_dtype, 1000)

    test_case_code = {}
    for index, test_case in enumerate(test_case_list):
        if data_type == 'int':
            formatted_input_value = str(test_case)
        elif data_type == 'list_int':
            formatted_input_value = str(test_case)
        elif data_type == 'string_of_int' or data_type == 'string':
            formatted_input_value = f"'{test_case}'" if isinstance(test_case, str) else str(test_case)

        modified_code = code.replace('test_input_value', formatted_input_value)
        test_case_code[index] = modified_code
    lineCoverage_dict[id] = test_case_code

# with open('./dataset/final_code_dataset.json', 'w') as file:
#     json.dump(lineCoverage_dict, file, indent=4)

with open('./dataset/final_Incomplete_Code_Dataset.json', 'w') as file:
    json.dump(lineCoverage_dict, file, indent=4)
