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

def remove_label(in_path, out_path):
    data = get_data(in_path)
    for item in data:
        item["entities"] = []
        item["relations"] = []
    output_data(data, out_path)

if __name__ == "__main__":
    # in_path = "../data/datasets/ade/ade_split_0_train.json"
    # out_path = "../data/datasets/ade/unlabeled_split_0.json"
    in_path = "../data/datasets/ade/unlabeled_all.json"
    out_path = "../data/datasets/ade/unlabeled_all.json"
    remove_label(in_path, out_path)