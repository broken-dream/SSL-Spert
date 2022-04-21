import json
import random
def cal_weight(cnt, alpha=1):
    cnt = [(k,v) for k,v in cnt.items()]
    cnt.sort(key=lambda x:x[1], reverse=True)
    res = dict()
    type_num = len(cnt)
    for i in range(type_num):
        res[cnt[i][0]] = pow(cnt[type_num-i-1][1] / cnt[0][1], alpha)
    return res

def get_prob(data_path, type_path):
    with open(type_path) as f:
        types = json.load(f)
    with open(data_path) as f:
        data = json.load(f)
    ner_cnt = dict()
    rel_cnt = dict()
    for ner in types["entities"]:
        ner_cnt[ner] = 0
    for rel in types["relations"]:
        rel_cnt[rel] = 0
    for item in data:
        for ner in item["entities"]:
            ner_cnt[ner["type"]] += 1
        for rel in item["relations"]:
            rel_cnt[rel["type"]] += 1
    
    ner_prob = cal_weight(ner_cnt)
    for k,v in ner_prob.items():
        print("{}:{}".format(k,v))
    rel_prob = cal_weight(rel_cnt)
    for k,v in rel_prob.items():
        print("{}:{}".format(k,v))
    return ner_prob, rel_prob

def set_sample_weight_ner(data, prob):
    for item in data:
        max_prob = 0
        for ner in item["entities"]:
            max_prob = max(max_prob, prob[ner["type"]])
        item["ner_prob"] = max_prob

def set_sample_weight_rel(data, prob):
    for item in data:
        max_prob = 0
        for rel in item["relations"]:
            max_prob = max(max_prob, prob[rel["type"]])
        item["rel_prob"] = max_prob

def sample_data_by_ner(data):
    res = []
    for item in data:
        prob = random.random()
        if prob < item["ner_prob"]:
            res.append(item)
    return res

def sample_data_by_rel(data):
    res = []
    for item in data:
        prob = random.random()
        if prob < item["rel_prob"]:
            res.append(item)
    return res
