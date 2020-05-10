import os
import random
import csv
# 0 - no tumour
# 1 - yes tumour

def partition_data(list_files, percent):
    n = int(round((percent/100)*len(list_files)))
    shuffled = list_files[:]
    random.shuffle(shuffled)
    return shuffled[n:], shuffled[:n]


def get_data_small():
    nb_classes = 2

no_tumour = []
yes_tumour = []

for root, dirs, files in os.walk('data-small/no/resized'):
    for file in files:
        p=os.path.join(root,file)
        no_tumour.append([p,"no"])

for root, dirs, files in os.walk('data-small/yes/resized'):
    for file in files:
        p=os.path.join(root,file)
        yes_tumour.append([p,"yes"])

test_no_tumour, train_no_tumour = partition_data(no_tumour,70)
test_yes_tumour, train_yes_tumour = partition_data(yes_tumour,70)

test = test_yes_tumour+test_no_tumour
test = random.sample(test, len(test))   

train = train_yes_tumour+train_no_tumour
train = random.sample(train, len(train))   

# Write to csv
with open("meta/train.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(train)

with open("meta/test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(test)
    