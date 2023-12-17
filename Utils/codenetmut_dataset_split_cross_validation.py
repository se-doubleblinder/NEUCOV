import json
import random

with open('../Dataset/codenetmut.json', 'r') as f:
    data = json.load(f)

with open('../Dataset/training.json', 'r') as f:
    training_data = json.load(f)
with open('../Dataset/testing.json', 'r') as f:
    testing_data = json.load(f)
with open('../Dataset/validation.json', 'r') as f:
    validation_data = json.load(f)

print(len(data))
print(len(training_data))
print(len(testing_data))
print(len(validation_data))

problem_statements = {
    1: (0, 629),
    2: (630, 947),
    3: (948, 1230),
    4: (1231, 1582),
    5: (1583, 2328),
    6: (2329, 2453),
    7: (2454, 2763),
    8: (2764, 3218),
    9: (3219, 3725),
    10: (3726, 3998),
    11: (3999, 4410),
    12: (4411, 5259),
    13: (5260, 6108),
    14: (6109, 6658),
    15: (6659, 7696),
    16: (7697, 8319),
    17: (8320, 8965),
    18: (8966, 9081),
    19: (9082, 9623),
    20: (9624, 10300),
    21: (10301, 10689),
    22: (10690, 11197),
    23: (11198, 11725),
    24: (11726, 12206),
    25: (12207, 12480),
    26: (12481, 12627),
    27: (12628, 13156),
    28: (13157, 13654),
    29: (13655, 13664),
    30: (13665, 13791),
    31: (13792, 14498),
    32: (14499, 15151),
    33: (15152, 15892),
    34: (15893, 16136),
    35: (16137, 16532),
    36: (16533, 17363),
    37: (17364, 18193),
    38: (18194, 18203),
    39: (18204, 18852),
    40: (18853, 18864),
    41: (18865, 19302),
    42: (19303, 19540)
}

total_instances = sum(end - start + 1 for start, end in problem_statements.values())

train_instances = int(0.8 * total_instances)
test_instances = validation_instances = int(0.1 * total_instances)

problem_list = list(problem_statements.items())

random.shuffle(problem_list)

train_set, test_set, validation_set = [], [], []
train_count, test_count, validation_count = 0, 0, 0

def create_balanced_subsets(problem_list, num_subsets=5):
    subsets = []

    for _ in range(num_subsets):
        random.shuffle(problem_list)
        train_set, test_set, validation_set = [], [], []
        train_count, test_count, validation_count = 0, 0, 0

        for problem, (start, end) in problem_list:
            problem_size = end - start + 1

            if train_count < 0.8 * total_instances:
                train_set.append(problem)
                train_count += problem_size
            elif test_count < 0.1 * total_instances:
                test_set.append(problem)
                test_count += problem_size
            elif validation_count < 0.1 * total_instances:
                validation_set.append(problem)
                validation_count += problem_size

        subsets.append((train_set, test_set, validation_set))

    return subsets

def calculate_instances(subset, problem_statements):
    total = 0
    for problem in subset:
        start, end = problem_statements[problem]
        total += end - start + 1
    return total

balanced_subsets = create_balanced_subsets(problem_list, num_subsets=5)

for index, subset in enumerate(balanced_subsets):
    train_set, test_set, validation_set = subset
    training_dataset = {}
    testing_dataset = {}
    validation_dataset = {}
    print(f"Training Subset: {train_set} , Train instances: {calculate_instances(train_set, problem_statements)}")
    print(f"Testing Subset: {test_set} , Test instances: {calculate_instances(test_set, problem_statements)}")
    print(f"Validation Subset: {validation_set} , Validation instances: {calculate_instances(validation_set, problem_statements)}")
    print()

    for i in train_set:
        start, end = problem_statements[i]
        for j in range(start, end+1):
            # training_dataset[int(data[str(j)]["id"])] = {"id": data[str(j)]["id"], "code": data[str(j)]["code"], "coverage": data[str(j)]["coverage"]}
            if str(data[str(j)]["id"]) in training_data:
                training_dataset[str(data[str(j)]["id"])] = training_data[str(data[str(j)]["id"])]
            elif str(data[str(j)]["id"]) in testing_data:
                training_dataset[str(data[str(j)]["id"])] = testing_data[str(data[str(j)]["id"])]
            elif str(data[str(j)]["id"]) in validation_data:
                training_dataset[str(data[str(j)]["id"])] = validation_data[str(data[str(j)]["id"])]


    for i in test_set:
        start, end = problem_statements[i]
        for j in range(start, end+1):
            # testing_dataset[int(data[str(j)]["id"])] = {"id": data[str(j)]["id"], "code": data[str(j)]["code"], "coverage": data[str(j)]["coverage"]}
            if str(data[str(j)]["id"]) in training_data:
                testing_dataset[str(data[str(j)]["id"])] = training_data[str(data[str(j)]["id"])]
            elif str(data[str(j)]["id"]) in testing_data:
                testing_dataset[str(data[str(j)]["id"])] = testing_data[str(data[str(j)]["id"])]
            elif str(data[str(j)]["id"]) in validation_data:
                testing_dataset[str(data[str(j)]["id"])] = validation_data[str(data[str(j)]["id"])]

    for i in validation_set:
        start, end = problem_statements[i]
        for j in range(start, end+1):
            # validation_dataset[int(data[str(j)]["id"])] = {"id": data[str(j)]["id"], "code": data[str(j)]["code"], "coverage": data[str(j)]["coverage"]}
            if str(data[str(j)]["id"]) in training_data:
                validation_dataset[str(data[str(j)]["id"])] = training_data[str(data[str(j)]["id"])]
            elif str(data[str(j)]["id"]) in testing_data:
                validation_dataset[str(data[str(j)]["id"])] = testing_data[str(data[str(j)]["id"])]
            elif str(data[str(j)]["id"]) in validation_data:
                validation_dataset[str(data[str(j)]["id"])] = validation_data[str(data[str(j)]["id"])]
    
    with open(f'../Dataset/dataset_{index}/training_{index}.json', 'w') as f:
        json.dump(training_dataset, f)
    
    with open(f'../Dataset/dataset_{index}/testing_{index}.json', 'w') as f:
        json.dump(testing_dataset, f)

    with open(f'../Dataset/dataset_{index}/validation_{index}.json', 'w') as f:
        json.dump(validation_dataset, f)