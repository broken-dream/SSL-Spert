from abc import ABC

import torch


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class SpERTLoss(Loss):
    def __init__(self, rel_criterion, entity_criterion, model, optimizer, scheduler, max_grad_norm):
        self._rel_criterion = rel_criterion
        self._entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm

    def compute(self, entity_logits, rel_logits, entity_types, rel_types, entity_sample_masks, rel_sample_masks):
        # entity loss
        # print("supervised labeled rel logits:{}".format(rel_logits))
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()

        entity_loss = self._entity_criterion(entity_logits, entity_types)
        entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

        # relation loss
        rel_sample_masks = rel_sample_masks.view(-1).float()
        rel_count = rel_sample_masks.sum()

        if rel_count.item() != 0:
            rel_logits = rel_logits.view(-1, rel_logits.shape[-1])
            rel_types = rel_types.view(-1, rel_types.shape[-1])

            rel_loss = self._rel_criterion(rel_logits, rel_types)
            rel_loss = rel_loss.sum(-1) / rel_loss.shape[-1]
            rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count

            # joint loss
            train_loss = entity_loss + rel_loss
        else:
            # corner case: no positive/negative relation samples
            train_loss = entity_loss

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()

class FixLoss(Loss):
    def __init__(self, rel_criterion, entity_criterion, model, optimizer, scheduler, max_grad_norm):
        self._rel_criterion = rel_criterion
        self._entity_criterion = entity_criterion
        # self._consistency_criterion = consistency_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm

    def compute(self, entity_logits, rel_logits, strong_entity_logits, strong_rel_logits, entity_types, rel_types, 
                entity_sample_masks, rel_sample_masks, entity_gold_masks, rel_gold_masks):
        # print("rel logits:{}".format(rel_logits[0][0]))
        # print("strong rel logits:{}".format(strong_rel_logits[0][0]))
        # entity loss
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()
        entity_gold_masks = entity_gold_masks.view(-1).float()

        entity_loss = self._entity_criterion(entity_logits, entity_types) 
        entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

        # strong entity logits
        strong_entity_logits = strong_entity_logits.view(-1, strong_entity_logits.shape[-1])
        strong_entity_loss = self._entity_criterion(strong_entity_logits, entity_types)
        strong_entity_loss = (strong_entity_loss * entity_gold_masks).sum() / entity_gold_masks.sum()

        # relation loss
        rel_sample_masks = rel_sample_masks.view(-1).float()
        rel_gold_masks = rel_gold_masks.view(-1).float()
        rel_count = rel_sample_masks.sum()
        rel_gold_count = rel_gold_masks.sum()

        if rel_count.item() != 0:
            rel_logits = rel_logits.view(-1, rel_logits.shape[-1])
            rel_types = rel_types.view(-1, rel_types.shape[-1])

            rel_loss = self._rel_criterion(rel_logits, rel_types)
            rel_loss = rel_loss.sum(-1) / rel_loss.shape[-1]
            rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count

            # strong relation loss
            strong_rel_logits = strong_rel_logits.view(-1, strong_rel_logits.shape[-1])
            # print("reviewed strong rel logits:{}".format(strong_rel_logits))
            # print("strong_rel_logits shape:{}".format(strong_rel_logits.shape))
            strong_rel_loss = self._rel_criterion(strong_rel_logits, rel_types)
            # print("reviewed strong rel loss:{}".format(strong_rel_loss))
            # print("strong criterion shape:{}".format(strong_rel_loss))
            strong_rel_loss = strong_rel_loss.sum(-1) / strong_rel_loss.shape[-1]
            # print("divided strong rel loss:{}".format(strong_rel_loss))
            if rel_gold_count.item() != 0:
                strong_rel_loss = (strong_rel_loss * rel_gold_masks).sum() / rel_gold_count
            else:
                strong_rel_loss = 0
            # print("gold mask sum:{}".format(rel_gold_masks.sum()))
            # print("final strong rel loss:{}".format(strong_rel_loss))
            # joint loss
            # train_loss = entity_loss + rel_loss + strong_entity_loss + strong_rel_loss
            
            # for debug
            train_loss = entity_loss + rel_loss + strong_entity_loss + strong_rel_loss
            # train_loss = entity_loss + rel_loss + strong_entity_loss
        else:
            # corner case: no positive/negative relation samples
            train_loss = entity_loss + strong_entity_loss
            # for debug
            # train_loss = entity_loss

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()