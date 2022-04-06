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

def sample(in_path, out_path, cnt):
    data = get_data(in_path)
    sampled_data = random.sample(data, cnt)
    output_data(sampled_data, out_path)

if __name__ == "__main__":
    total = {
        "scierc":1861,
        "ade":3395,
        "scierc_1":1681
    }
    dataset = "scierc_1"
    # in_path = "../data/datasets/scierc/labeled_all.json"
    # out_path = "../data/datasets/scierc/labeled-100.json"
    in_path = "../data/datasets/{}/labeled_all.json".format(dataset)
    out_path = "../data/datasets/{}/labeled-30.json".format(dataset)
    ratio = 0.3
    cnt = int(total[dataset]*ratio)
    sample(in_path, out_path, cnt)