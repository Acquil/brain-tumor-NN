import os
import random
import csv
# 0 - no tumour
# 1 - yes tumour

data = "data-original"

def partition_data(list_files, percent):
    n = int(round((percent/100)*len(list_files)))
    shuffled = list_files[:]
    random.shuffle(shuffled)
    return shuffled[n:], shuffled[:n]

no_tumour = []
yes_tumour = []

for root, dirs, files in os.walk('{0}/no/resized'.format(data)):
    for file in files:
        p=os.path.join(root,file)
        no_tumour.append([p,0])

for root, dirs, files in os.walk('{0}/yes/resized'.format(data)):
    for file in files:
        p=os.path.join(root,file)
        yes_tumour.append([p,1])

test_no_tumour, train_no_tumour = partition_data(no_tumour,70)
test_yes_tumour, train_yes_tumour = partition_data(yes_tumour,70)

test = test_yes_tumour+test_no_tumour
test = random.sample(test, len(test))   

train = train_yes_tumour+train_no_tumour
train = random.sample(train, len(train))   

# Write to csv
with open("meta/{0}/train.csv".format(data), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(train)

with open("meta/{0}/test.csv".format(data), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(test)
    