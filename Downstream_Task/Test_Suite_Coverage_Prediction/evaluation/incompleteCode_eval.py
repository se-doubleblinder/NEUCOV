# Least 5 lines
macro_accuracy_list_least = []
import json
with open('../dataset/IncompleteCode_ground_truth/output_without_importline.json', 'r') as file:
    data = json.load(file)

for id in data:
    # Ground Truth
    ground_truth_id = id
    ground_truth_coverage = []

    for tId in data[id]:
        ground_truth_coverage.append(data[id][tId]['execution_array'])

    with open(f'../output/model_partialDataset_output/partialCode_output_downstream_task_{id}.json' , 'r') as file:
        model_output = json.load(file)

     # Predicted
    pred_coverage = []
    for index in model_output:
        for pred in index:
            pred_coverage.append(pred)

    # Getting more Errors on Ground Truth as compare to the Model Output
    if len(ground_truth_coverage) < len(pred_coverage):
        merged_ground_truth_coverage = {}
        merged_pred_coverage = {}
        for tId in data[id]:
            current_ground_truth = data[id][tId]['execution_array']
            current_pred = pred_coverage[int(tId)]

            for i in range(len(current_ground_truth)):
                if current_ground_truth[i] == 1:
                    if i in merged_ground_truth_coverage:
                        merged_ground_truth_coverage[i] += 1
                    else:
                        merged_ground_truth_coverage[i] = 1
                else:
                    merged_ground_truth_coverage[i] = 0

            for i in range(len(current_pred)):     
                if current_pred[i] == 1:
                    if i in merged_pred_coverage:
                        merged_pred_coverage[i] += 1
                    else:
                        merged_pred_coverage[i] = 1
                else:
                    merged_pred_coverage[i] = 0
            
        sorted_gt_dict_asc = dict(sorted(merged_ground_truth_coverage.items(), key=lambda item: item[1]))
        sorted_pd_dict_asc = dict(sorted(merged_pred_coverage.items(), key=lambda item: item[1]))
        
        sorted_gt_dict_desc = dict(sorted(merged_ground_truth_coverage.items(), key=lambda item: item[1], reverse=True))
        sorted_pd_dict_desc = dict(sorted(merged_pred_coverage.items(), key=lambda item: item[1], reverse=True))

        # Macro Accuracy Least
        count = 0
        for i in range(5):
            if sorted_gt_dict_asc[i] == sorted_pd_dict_asc[i]:
                count += 1
            macro_accuracy_list_least.append((count/5)*100)

    # Getting same Model Output as compare to the Ground Truth
    if len(ground_truth_coverage) == len(pred_coverage):
        # Merge Line number Sum 
        merged_ground_truth_coverage = {}

        for i in range(len(ground_truth_coverage)):
            for j in range(len(ground_truth_coverage[i])):
                if ground_truth_coverage[i][j] == 1:
                    if j in merged_ground_truth_coverage:
                        merged_ground_truth_coverage[j] += 1
                    else:
                        merged_ground_truth_coverage[j] = 1
                else:
                    merged_ground_truth_coverage[j] = 0
        
        # Merge preds Line number Sum
        merged_pred_coverage = {}

        for pred in pred_coverage:      
            for i in range(len(pred)):
                if pred[i] == 1:
                    if i in merged_pred_coverage:
                        merged_pred_coverage[i] += 1
                    else:
                        merged_pred_coverage[i] = 1
                else:
                    merged_pred_coverage[i] = 0
        
        # Ascending Order
        sorted_gt_dict_asc = dict(sorted(merged_ground_truth_coverage.items(), key=lambda item: item[1]))
        sorted_pd_dict_asc = dict(sorted(merged_pred_coverage.items(), key=lambda item: item[1]))
        # Descending Order
        sorted_gt_dict_desc = dict(sorted(merged_ground_truth_coverage.items(), key=lambda item: item[1], reverse=True))
        sorted_pd_dict_desc = dict(sorted(merged_pred_coverage.items(), key=lambda item: item[1], reverse=True))

        # Macro Accuracy Least
        count = 0
        for i in range(5):
            if sorted_gt_dict_asc[i] == sorted_pd_dict_asc[i]:
                count += 1
        macro_accuracy_list_least.append((count/5)*100)
        
# Macro Acuracy
print(f"Length of the Macro List Least: {len(macro_accuracy_list_least)}")
print(f"Macro Accuracy Least: {sum(macro_accuracy_list_least) / len(macro_accuracy_list_least)}")

print("="*50)
