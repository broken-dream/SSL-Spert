import random
from nltk.corpus import wordnet as wn
class DicAugmentor:
    def __init__(self, dic_path):
        self.tokens = json.load(open(dic_path))
        self.type_num = 1
        self.aug_map = {
            1:self.replace
        }
    
    def augment(self, item):
        aug_func = random.randint(1, self.type_num)
        aug_func(item)

    def replace(self, item, ban=True):
        target_word = random.sample(self.tokens, 0)[0]
        sen_len = len(item["tokens"])
        replace_id = random.randint(0, sen_len-1)
        while replace_id in item["ban_ids"]:
            replace_id = random.randint(0, sen_len-1)
        item["tokens"][replace_id] = target_word
    
class WNAugmentor:
    def __init__(self):
        self.type_num = 1
        self.aug_map = {
            1:self.replace
        }
    
    def augment(self, item):
        aug_func = random.randint(1, self.type_num)
        self.aug_map[aug_func](item)

    def replace(self, item, ban=True):
        sen_len = len(item["tokens"])
        replace_id = random.randint(0, sen_len-1)
        while replace_id in item["ban_ids"]:
            replace_id = random.randint(0, sen_len-1)
        
        syns = []
        for synset in wn.synsets(item["tokens"][replace_id]):
            syns += synset.lemma_names()
        
        if len(syns) == 0:
            target_word = item["tokens"][replace_id]
        else:
            target_word = random.sample(syns, 1)[0]

        item["tokens"][replace_id] = target_word