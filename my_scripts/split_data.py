import random
import json
def get_data(file_path):
    f = open(file_path, encoding="utf-8")
    data = json.load(f)
    f.close()
    return data

def output_data(data, target_path):
    f = open(target_path, "w+", encoding="utf-8")
    json.dump(data, f)
    f.close()

def split_data(in_path, out_path1, out_path2):
    data = get_data(in_path)
    data1 = []
    data2 = []
    for item in data:
        if random.random() < 0.5:
            data1.append(item)
        else :
            data2.append(item)
    print("{}:{}".format(len(data1), len(data2)))
    output_data(data1, out_path1)
    output_data(data2, out_path2)

if __name__ == "__main__":
    in_path = "../data/datasets/ade/train.json"
    out_path1 = "../data/datasets/ade/labeled_all.json"
    out_path2 = "../data/datasets/ade/unlabeled_all.json"
    split_data(in_path, out_path1, out_path2)