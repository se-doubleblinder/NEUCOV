import json

with open ('./ground_truth_data_5.json', 'r') as file:
    data = json.load(file)
with open ('./ground_truth_data_10.json', 'r') as file:
    data2 = json.load(file)
with open ('./ground_truth_data_15.json', 'r') as file:
    data3 = json.load(file)

finalData = {}
finalData.update(data)
finalData.update(data2)
finalData.update(data3)

with open('../raw_incomplete_code.json', 'r') as file:
    rawData = json.load(file)

with open('../final_Incomplete_Code_Dataset.json', 'r') as file:
    data_with_testCase = json.load(file)

final_without_import_line_code = {}
for id in finalData:
    tId_data = {}
    for tId in finalData[id]:
        execution_array = finalData[id][tId]['execution_array']
        line_to_remove = rawData[id]['import_line']
        execution_array.pop(line_to_remove)
        tId_data[tId] = {'code' : data_with_testCase[id][tId], 'execution_array' : execution_array}
    final_without_import_line_code[id] = tId_data



with open('output_without_importline.json', 'w') as file:
    json.dump(final_without_import_line_code, file, indent=4)