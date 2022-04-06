import json

def get_data(file_path):
    f = open(file_path, encoding="utf-8")
    data = json.load(f)
    f.close()
    return data

data1 = get_data("../data/datasets/ade/unlabeled_predictions.json")
data2 = get_data("../data/datasets/scierc/labeled-30.json")
print("{}:{}".format(len(data1), len(data2)))