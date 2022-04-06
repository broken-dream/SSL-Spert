import json
def get_data(file_path):
    f = open(file_path, encoding="utf-8")
    data = json.load(f)
    f.close()
    return data

def merge_data(golds, preds):
    final_data = []
    for gold, pred in zip(golds, preds):
        cur = dict()
        cur["tokens"] = gold["tokens"]
        cur["gold_entities"] = gold["entities"]
        cur["gold_relations"] = gold["relations"]
        cur["pred_entities"] = pred["entities"]
        cur["pred_relations"] = pred["relations"]
        final_data.append(cur)
    return final_data

def get_gold(gold, unlabeled):
    res = []
    gold_map = dict()
    for item in gold:
        gold_map[item["orig_id"]] = item
    for item in unlabeled:
        res.append(gold_map[item["orig_id"]])
    return res

def output_data(data, target_path):
    f = open(target_path, "w+", encoding="utf-8")
    json.dump(data, f)
    f.close()

gold_data = get_data("../data/datasets/scierc_1/train.json")
unlabeled_data = get_data("../data/datasets/scierc_1/unlabeled_all.json")
gold_data = get_gold(gold_data, unlabeled_data)
output_data(gold_data, "../data/datasets/scierc_1/unlabeled_init.json")
