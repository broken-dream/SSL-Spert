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

def mask_ent(in_path, out_path):
    data = get_data(in_path)
    for item in data:
        for ner in item["entities"]:
            for i in range(ner["start"], ner["end"]):
                item["tokens"][i] = "[UNK]"
    output_data(data, out_path)

if __name__ == "__main__":
    # in_path = "../data/datasets/scierc/dev.json"
    # out_path = "../data/datasets/masked_scierc/dev.json"
    in_path = "../data/datasets/ade/ade_split_0_test.json"
    out_path = "../data/datasets/masked_ade/ade_split_0_test.json"
    mask_ent(in_path, out_path)