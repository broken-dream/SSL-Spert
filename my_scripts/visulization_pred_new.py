import json

def get_data(file_path):
    f = open(file_path, encoding="utf-8")
    data = json.load(f)
    f.close()
    return data

def get_data_line(file_path):
    res = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if len(line) == 0:
                continue
            res.append(json.loads(line))
    return res

def output_data(data, target_path):
    pass

def merge_data(golds, preds):
    final_data = []
    for pred in preds:
        for gold in golds:
            if pred["tokens"] == gold["tokens"]:
                cur = dict()
                cur["tokens"] = gold["tokens"]
                cur["gold_entities"] = gold["entities"]
                cur["gold_relations"] = gold["relations"]
                cur["pred_entities"] = pred["entities"]
                cur["pred_relations"] = pred["relations"]
                final_data.append(cur)
    return final_data

def token2str(tokens):
    res = tokens[0]
    for i in range(1, len(tokens)):
        res += " " + tokens[i]
    return res

def visualization_ner_re(tokens, entities, relations):
    ent_map = dict()
    for i in range(len(entities)):
        ent_map[i] = (token2str(tokens[entities[i]["start"]:entities[i]["end"]]), entities[i]["type"])
    re_res = []
    for trip in relations:
        head_str = ent_map[trip["head"]][0]
        tail_str = ent_map[trip["tail"]][0]
        re_res.append((head_str, trip["type"], tail_str))
    return ent_map, re_res

def print_ner(ners, out_file):
    for k,v in ners.items():
        print("{}--{}".format(v[0], v[1]), file=out_file)

def print_re(relations, out_file):
    for item in relations:
        print("{}--{}--{}".format(item[0], item[1], item[2]), file=out_file)

def visualization(data, out_path):
    f = open(out_path, "w+", encoding="utf-8")
    for item in data:
        gold_ent, gold_rel = visualization_ner_re(item["tokens"], item["gold_entities"], item["gold_relations"])
        pred_ent, pred_rel = visualization_ner_re(item["tokens"], item["pred_entities"], item["pred_relations"])
        print(token2str(item["tokens"]), file=f)
        print("#######entities#######", file=f)
        print_ner(gold_ent, out_file=f)
        print("----------------------", file=f)
        print_ner(pred_ent, out_file=f)
        print("#######relations#######", file=f)
        print_re(gold_rel, out_file=f)
        print("----------------------", file=f)
        print_re(pred_rel, out_file=f)
        print("*********************************************", file=f)
        print("", file=f)
    f.close()

def get_gold(gold, unlabeled):
    res = []
    gold_map = dict()
    for item in gold:
        gold_map[item["orig_id"]] = item
    for item in unlabeled:
        res.append(gold_map[item["orig_id"]])
    return res
        


if __name__ == "__main__":
    dataset_name = "scierc"
    gold_data = get_data("../data/datasets/{}/train.json".format(dataset_name))
    unlabeled_data = get_data("../data/datasets/{}/unlabeled_all.json".format(dataset_name))
    pred_data = get_data_line("../data/datasets/{}/unlabeled_predictions_2.json".format(dataset_name))
    gold_data = get_gold(gold_data, unlabeled_data)
    merged_data = merge_data(gold_data, pred_data)
    visualization(merged_data, "../data/datasets/{}/visualization_2.json".format(dataset_name))