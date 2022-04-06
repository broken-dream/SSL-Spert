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

def get_ner_map(data):
    res = dict()
    span_map = dict()
    for i in range(len(data)):
        span = (data[i]["start"], data[i]["end"])
        res[i] = (span, data[i]["type"])
        span_map[span] = data[i]["type"]
    return res, span_map

def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0


def compute_f1(predicted, gold, matched):
    # F1 score.
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return dict(precision=precision, recall=recall, f1=f1)

def evaluate(data):
    ner_data = {
        "gold":0,
        "pred":0,
        "matched":0
    }
    rel_data = {
        "gold":0,
        "pred":0,
        "matched":0,
        "matched_ner":0
    }
    for item in data:
        gold_entities = item["gold_entities"]
        pred_entities = item["pred_entities"]
        gold_relations = item["gold_relations"]
        pred_relations = item["pred_relations"]
        ner_data["gold"] += len(gold_entities)
        ner_data["pred"] += len(pred_entities)
        rel_data["gold"] += len(gold_relations)
        rel_data["pred"] += len(pred_relations)
        gold_map, gold_span_map = get_ner_map(gold_entities)
        pred_map, pred_span_map = get_ner_map(pred_entities)

        gold_rel_map = dict()
        for rel in gold_relations:
            head_span = gold_map[rel["head"]][0]
            tail_span = gold_map[rel["tail"]][0]
            gold_rel_map[(head_span, tail_span)] = rel["type"]

        # cal ner
        for ner in pred_entities:
            if any([ner == gold for gold in gold_entities]):
                ner_data["matched"] += 1
        
        for rel in pred_relations:
            head_span = pred_map[rel["head"]][0]
            tail_span = pred_map[rel["tail"]][0]
            if gold_rel_map.get((head_span, tail_span), None) == rel["type"]:
                rel_data["matched"] += 1
                if pred_span_map[head_span] == gold_span_map[head_span] and \
                   pred_span_map[tail_span] == gold_span_map[tail_span]:
                   rel_data["matched_ner"] += 1
    
    res = compute_f1(ner_data["pred"], ner_data["gold"], ner_data["matched"])
    print("NER: precision:{}  recall:{}  f1:{}".format(res["precision"],res["recall"],res["f1"]))
    res = compute_f1(rel_data["pred"], rel_data["gold"], rel_data["matched"])
    print("REL: precision:{}  recall:{}  f1:{}".format(res["precision"],res["recall"],res["f1"]))
    res = compute_f1(rel_data["pred"], rel_data["gold"], rel_data["matched"])
    print("REL+: precision:{}  recall:{}  f1:{}".format(res["precision"],res["recall"],res["f1"]))

if __name__ == "__main__":
    gold_data = get_data("../data/datasets/scierc/train.json")
    unlabeled_data = get_data("../data/datasets/scierc/unlabeled_all.json")
    gold_data = get_gold(gold_data, unlabeled_data)
    pred_data = get_data("../data/datasets/scierc/unlabeled_predictions.json")
    merged_data = merge_data(gold_data, pred_data)
    evaluate(merged_data)
