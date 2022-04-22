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

def split_by_ner(data, prob):
    res = dict()
    for k in prob:
        res[k] = list()

    for item in data:
        max_prob = 0
        item["ner_prob"] = 0
        item["sig_ner"] = "no_type"
        for ner in item["entities"]:
            if prob[ner["type"]] > item["ner_prob"]:
                item["ner_prob"] = prob[ner["type"]]
                item["sig_ner"] = ner["type"]
        if item["sig_ner"] != "no_type":
            res[item["sig_ner"]].append(item)
    
    return res

def split_by_rel(data, prob):
    res = dict()
    for k in prob:
        res[k] = list()

    for item in data:
        item["rel_prob"] = 0
        item["sig_rel"] = "no_type"
        for rel in item["relations"]:
            if prob[rel["type"]] > item["rel_prob"]:
                item["rel_prob"] = prob[rel["type"]]
                item["sig_rel"] = rel["type"]
        if item["sig_rel"] != "no_type":
            res[item["sig_rel"]].append(item)
    
    return res

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

def sample_data_crest(data, prob, alpha):
    res = []
    for k,v in data.items():
        cnt = len(v) * pow(prob[k], 1/alpha)
        res += v[:cnt]
    return res