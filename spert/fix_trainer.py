import argparse
import math
import os
from typing import Type

import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
import transformers
from torch.utils.data import DataLoader
from transformers import AdamW, BertConfig
from transformers import BertTokenizer

from spert import models, prediction
from spert import sampling
from spert import util
from spert.entities import Dataset, TwoStreamDataset
from spert.evaluator import Evaluator
from spert.input_reader import JsonInputReader, BaseInputReader, AugInputReader
from spert.loss import SpERTLoss, Loss, FixLoss
from tqdm import tqdm
from spert.trainer import BaseTrainer

from spert import balancing
from spert import augmentor

import json
import functools

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class SpERTTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # byte-pair encoding
        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                        do_lower_case=args.lowercase,
                                                        cache_dir=args.cache_path)
        self.model = None

        self.best_metric = {
            "ner":0,
            "re":0,
            "re+":0
        }
        self.best_epoch = 0
        self.global_iteration = 0

    def train(self, train_path: str, valid_path: str, types_path: str, unlabeled_path: str, input_reader_cls: Type[BaseInputReader]):
        # refresh predictions file
        with open(self._args.unlabeled_predictions_path, 'w+') as predictions_file:
            print("", end="", file=predictions_file)
        
        ner_prob, rel_prob = balancing.get_prob(self._args.train_path, self._args.types_path)

        args = self._args
        train_label, valid_label = 'train', 'valid'
        unlabeled_label = 'unlabeled'
        weak_label = 'weak'
        strong_label = 'strong'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("Model type: %s" % args.model_type)

        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # read datasets
        cur_augmentor = augmentor.WNAugmentor()
        input_reader = input_reader_cls(types_path, self._tokenizer, args.neg_entity_count,
                                        args.neg_relation_count, args.max_span_size, self._logger, cur_augmentor)
        input_reader.read(train_path, train_label)
        input_reader.read(valid_path, valid_label)
        input_reader.read(unlabeled_path, unlabeled_label)
        train_dataset = input_reader.gen_dataset(train_label)
        validation_dataset = input_reader.gen_dataset(valid_label)
        unlabeled_dataset = input_reader.gen_dataset(unlabeled_label)

        all_predictions = []
        self._log_datasets(input_reader)

        semi_epochs = args.semi_end_epoch - args.semi_epoch
        semi_cnt = len(unlabeled_dataset)

        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // args.train_batch_size
        updates_total = updates_epoch * args.epochs
        semi_total = (semi_cnt//args.train_batch_size)*semi_epochs
        if args.semi:
            # batch num for semi epochs
            semi_total = (semi_cnt//args.train_batch_size)*semi_epochs
            if args.unlabeled_type == "ner_rel" and not args.semi_cnt_control:
                semi_total *= 2
            print("{}:{}".format(updates_total, semi_total))

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        # load model
        model = self._load_model(input_reader)
        
        if args.transfer:
            model.load_src(args.src_path, start_layer=10)

        # SpERT is currently optimized on a single GPU and not thoroughly tested in a multi GPU setup
        # If you still want to train SpERT on multiple GPUs, uncomment the following lines
        # # parallelize model
        # if self._device.type != 'cpu':
        #     model = torch.nn.DataParallel(model)

        model.to(self._device)


        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        # create scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total+semi_total)
        # create loss function
        rel_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        compute_loss = SpERTLoss(rel_criterion, entity_criterion, model, optimizer, scheduler, args.max_grad_norm)
        fix_loss = FixLoss(rel_criterion, entity_criterion, model, optimizer, scheduler, args.max_grad_norm)
        # eval validation set
        if args.init_eval:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)
        
        
        # train
        for epoch in range(args.epochs):
            # train epoch
            self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch)
            
            if args.fix and epoch >= args.semi_epoch and epoch <args.semi_end_epoch:
                if args.unlabeled_type == "default":
                    predictions = self._predict_unlabeled(model, unlabeled_dataset, input_reader, semi_cnt)
                elif args.unlabeled_type == "ner_rel":
                    predictions = self._predict_unlabeled_balancing_ner_rel(model, unlabeled_dataset, input_reader, semi_cnt, ner_prob, rel_prob, args.semi_cnt_control, args.sort_state)
                else:
                    predictions = self._predict_unlabeled_balancing(model, unlabeled_dataset, input_reader, semi_cnt, ner_prob, rel_prob, args.unlabeled_type, args.sort_state)
                print("unlabeled data for train:{}".format(len(predictions)))
                if len(predictions) > 0: 
                    input_reader.create_documents(weak_label, predictions)
                    weak_dataset = input_reader.gen_dataset(weak_label)
                    strong_dataset = input_reader.gen_dataset(weak_label, aug=True)
                    two_stream_dataset = TwoStreamDataset(weak_dataset, strong_dataset)
                    
                    # for debug
                    # f = open("/data2/wh/spert/data/datasets/scierc_wo_generic/pseudo_weak1.json", "w+")
                    # f1 = open("/data2/wh/spert/data/datasets/scierc_wo_generic/pseudo_strong1.json", "w+")
                    # data_loader = DataLoader(two_stream_dataset, batch_size=self._args.train_batch_size, shuffle=True, drop_last=True,
                    #                             num_workers=self._args.sampling_processes, collate_fn=sampling.two_stream_collate_fn_padding)
                    # for b_idx, batch in enumerate(data_loader):
                    #     for i in range(self._args.train_batch_size):
                    #         print(b_idx*self._args.train_batch_size+i,file=f)
                    #         print(input_reader._tokenizer.convert_ids_to_tokens(batch["encodings"][i]), file=f)
                    #         for k,v in batch.items():
                    #             print("{}:{}".format(k,v[i]), file=f)
                    #         print("", file=f)
                    #         print(b_idx*self._args.train_batch_size+i,file=f1)
                    #         print(input_reader._tokenizer.convert_ids_to_tokens(batch["encodings"][i+self._args.train_batch_size]), file=f1)
                    #         for k,v in batch.items():
                    #             print("{}:{}".format(k,v[i+self._args.train_batch_size]), file=f1)
                    #         print("", file=f1)

                    self._train_fix_epoch(model, fix_loss, optimizer, two_stream_dataset, updates_epoch, epoch, True)

            # for debug
            # if epoch == args.semi_end_epoch:
            #     f = open("/data2/wh/spert/data/datasets/scierc_wo_generic/pseudo_weak.json", "w+")
            #     f1 = open("/data2/wh/spert/data/datasets/scierc_wo_generic/pseudo_strong.json", "w+")
            #     weak_dataset = input_reader.gen_dataset(train_label)
            #     strong_dataset = input_reader.gen_dataset(train_label, aug=True)
            #     two_stream_dataset = TwoStreamDataset(weak_dataset, strong_dataset)
            #     data_loader = DataLoader(two_stream_dataset, batch_size=self._args.train_batch_size, shuffle=True, drop_last=True,
            #                                 num_workers=self._args.sampling_processes, collate_fn=sampling.two_stream_collate_fn_padding)
            #     for batch in data_loader:
            #         for i in range(self._args.train_batch_size):
            #             print(input_reader._tokenizer.convert_ids_to_tokens(batch["encodings"][i]), file=f)
            #             for k,v in batch.items():
            #                 print("{}:{}".format(k,v[i]), file=f)
            #             print("", file=f)
            #             print(input_reader._tokenizer.convert_ids_to_tokens(batch["encodings"][i+self._args.train_batch_size]), file=f1)
            #             for k,v in batch.items():
            #                 print("{}:{}".format(k,v[i+self._args.train_batch_size]), file=f1)
            #             print("", file=f1)

                            

            # eval validation sets
            if not args.final_eval or (epoch == args.epochs - 1):
                eval_res = self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)
                eval_ave = (eval_res["ner"] + eval_res["re"] + eval_res["re+"]) / 3
                best_ave = (self.best_metric["ner"] + self.best_metric["re"] + self.best_metric["re+"]) / 3
                if eval_ave > best_ave:
                    best_ave = eval_ave
                    self.best_epoch = epoch
                    extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
                    global_iteration = args.epochs * updates_epoch
                    self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                                     optimizer=optimizer if self._args.save_optimizer else None, extra=extra,
                                     include_iteration=False, name='final_model')
                    self.best_epoch = epoch
                    self.best_metric = eval_res
                    print("best checkpoint:{}".format(epoch))
         
        self._logger.info("Logged in: %s" % self._log_path)
        self._logger.info("Saved in: %s" % self._save_path)
        self._close_summary_writer()
        print("----------------------------------")
        for k in self.best_metric:
            print("best {} : {}".format(k, self.best_metric[k]))
        print("best epoch:{}".format(self.best_epoch))
        print("----------------------------------")

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args
        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer,
                                        max_span_size=args.max_span_size, logger=self._logger)
        test_dataset = input_reader.read(dataset_path, dataset_label)
        self._log_datasets(input_reader)

        # load model
        model = self._load_model(input_reader)
        model.to(self._device)

        # evaluate
        self._eval(model, test_dataset, input_reader)

        self._logger.info("Logged in: %s" % self._log_path)
        self._close_summary_writer()

    def predict(self, dataset_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer,
                                        max_span_size=args.max_span_size,
                                        spacy_model=args.spacy_model)
        dataset = input_reader.read(dataset_path, 'dataset')

        model = self._load_model(input_reader)
        model.to(self._device)

        self._predict(model, dataset, input_reader)

    def _load_model(self, input_reader):
        model_class = models.get_model(self._args.model_type)

        config = BertConfig.from_pretrained(self._args.model_path, cache_dir=self._args.cache_path)
        util.check_version(config, model_class, self._args.model_path)

        config.spert_version = model_class.VERSION
        model = model_class.from_pretrained(self._args.model_path,
                                            config=config,
                                            # SpERT model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            relation_types=input_reader.relation_type_count - 1,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self._args.max_pairs,
                                            prop_drop=self._args.prop_drop,
                                            size_embedding=self._args.size_embedding,
                                            freeze_transformer=self._args.freeze_transformer,
                                            cache_dir=self._args.cache_path)

        return model

    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int, train_unlabel: bool = False):
        self._logger.info("Train epoch: %s" % epoch)

        # create data loader
        dataset.switch_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(dataset, batch_size=self._args.train_batch_size, shuffle=True, drop_last=True,
                                 num_workers=self._args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        model.zero_grad()

        iteration = 0
        total = dataset.document_count // self._args.train_batch_size
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            model.train()
            # for k in batch:
                # print("{}:{}".format(k, batch[k].shape))
            batch = util.to_device(batch, self._device)
            # print(batch["encodings"][0][:15])
            # forward step
            entity_logits, rel_logits = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                                              entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                                              relations=batch['rels'], rel_masks=batch['rel_masks'])
            

            # compute loss and optimize parameters
            batch_loss = compute_loss.compute(entity_logits=entity_logits, rel_logits=rel_logits,
                                              rel_types=batch['rel_types'], entity_types=batch['entity_types'],
                                              entity_sample_masks=batch['entity_sample_masks'],
                                              rel_sample_masks=batch['rel_sample_masks'])


            # logging
            if not train_unlabel:
                self.global_iteration += self._args.train_batch_size
                iteration += 1
                global_iteration = epoch * updates_epoch + iteration

                # if global_iteration % self._args.train_log_iter == 0:
                    # self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)
                if self.global_iteration % (3*self._args.train_batch_size) == 0:
                    self._log_train(optimizer, batch_loss, epoch, iteration, self.global_iteration, dataset.label)
        print("current epoch loss:{}".format(batch_loss))
        return iteration

    def _train_fix_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer: Optimizer, dataset: Dataset,
                         updates_epoch: int, epoch: int, train_unlabel: bool = False):
        self._logger.info("Train unlabeled epoch: %s" % epoch)

        # create data loader
        dataset.switch_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(dataset, batch_size=self._args.train_batch_size, shuffle=True, drop_last=True,
                                 num_workers=self._args.sampling_processes, collate_fn=sampling.two_stream_collate_fn_padding)

        model.zero_grad()

        iteration = 0
        total = dataset.document_count // self._args.train_batch_size
        batch_loss = 0
        cur = 0
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            model.train()
            batch = util.to_device(batch, self._device)
            # for k in batch:
                # print("{}:{}".format(k, batch[k].shape))
            # forward step
            entity_logits, rel_logits = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                                              entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                                              relations=batch['rels'], rel_masks=batch['rel_masks'])
            
            weak_entity_logits, strong_entity_logits = entity_logits.chunk(2)
            weak_rel_logits, strong_rel_logits = rel_logits.chunk(2)
            # print("strong rel shape:{}".format(strong_rel_logits.shape))
            # print("rel god mask shape:{}".format(batch['rel_gold_masks'][:weak_entity_logits.shape[0]].shape))

            # compute loss and optimize parameters
            # print(cur)
            # cur += self._args.train_batch_size
            batch_loss = 0
            batch_loss = compute_loss.compute(entity_logits=weak_entity_logits, rel_logits=weak_rel_logits,
                                              strong_entity_logits=strong_entity_logits, strong_rel_logits=strong_rel_logits,
                                              rel_types=batch['rel_types'][:weak_entity_logits.shape[0]], 
                                              entity_types=batch['entity_types'][:weak_entity_logits.shape[0]],
                                              entity_sample_masks=batch['entity_sample_masks'][:weak_entity_logits.shape[0]],
                                              rel_sample_masks=batch['rel_sample_masks'][:weak_entity_logits.shape[0]],
                                              entity_gold_masks=batch['ent_gold_masks'][:weak_entity_logits.shape[0]],
                                              rel_gold_masks=batch['rel_gold_masks'][:weak_entity_logits.shape[0]])
            # print("train_unlabel:{}".format(train_unlabel))
            # logging
            if not train_unlabel:
                self.global_iteration += self._args.train_batch_size
                iteration += 1
                global_iteration = epoch * updates_epoch + iteration

                # if global_iteration % self._args.train_log_iter == 0:
                    # self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)
                if self.global_iteration % (3*self._args.train_batch_size) == 0:
                    self._log_train(optimizer, batch_loss, epoch, iteration, self.global_iteration, dataset.label)
        print("current epoch loss:{}".format(batch_loss))
        return iteration

    def _eval(self, model: torch.nn.Module, dataset: Dataset, input_reader: BaseInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        self._logger.info("Evaluate: %s" % dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator
        predictions_path = os.path.join(self._log_path, f'predictions_{dataset.label}_epoch_{epoch}.json')
        examples_path = os.path.join(self._log_path, f'examples_%s_{dataset.label}_epoch_{epoch}.html')
        evaluator = Evaluator(dataset, input_reader, self._tokenizer,
                              self._args.rel_filter_threshold, self._args.no_overlapping, predictions_path,
                              examples_path, self._args.example_count)

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self._args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self._args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self._args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                               entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                               entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                               inference=True)
                entity_clf, rel_clf, rels = result

                # evaluate batch
                evaluator.eval_batch(entity_clf, rel_clf, rels, batch)

        global_iteration = epoch * updates_epoch + iteration
        ner_eval, rel_eval, rel_nec_eval = evaluator.compute_scores()
        self._log_eval(*ner_eval, *rel_eval, *rel_nec_eval,
                       epoch, iteration, global_iteration, dataset.label)

        if self._args.store_predictions and not self._args.no_overlapping:
            evaluator.store_predictions()

        if self._args.store_examples:
            evaluator.store_examples()
        # print(ner_eval)
        # print(rel_eval)
        # print(rel_nec_eval)
        eval_res = {
            "ner": ner_eval[2],
            "re": rel_eval[2],
            "re+": rel_nec_eval[2]
        }
        return eval_res

    def _predict(self, model: torch.nn.Module, dataset: Dataset, input_reader: BaseInputReader):
        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self._args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self._args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        pred_entities = []
        pred_relations = []

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self._args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Predict'):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                               entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                               entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                               inference=True)
                entity_clf, rel_clf, rels = result

                # convert predictions
                predictions = prediction.convert_predictions(entity_clf, rel_clf, rels,
                                                             batch, self._args.rel_filter_threshold,
                                                             input_reader)

                batch_pred_entities, batch_pred_relations = predictions
                pred_entities.extend(batch_pred_entities)
                pred_relations.extend(batch_pred_relations)

        prediction.store_predictions(dataset.documents, pred_entities, pred_relations, self._args.predictions_path)

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self._args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self._args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,

                  rel_prec_micro: float, rel_rec_micro: float, rel_f1_micro: float,
                  rel_prec_macro: float, rel_rec_macro: float, rel_f1_macro: float,

                  rel_nec_prec_micro: float, rel_nec_rec_micro: float, rel_nec_f1_micro: float,
                  rel_nec_prec_macro: float, rel_nec_rec_macro: float, rel_nec_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_prec_micro', rel_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_micro', rel_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_micro', rel_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_prec_macro', rel_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_macro', rel_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_macro', rel_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_nec_prec_micro', rel_nec_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_micro', rel_nec_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_micro', rel_nec_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_prec_macro', rel_nec_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_macro', rel_nec_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_macro', rel_nec_f1_macro, global_iteration)

        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,

                      rel_prec_micro, rel_rec_micro, rel_f1_micro,
                      rel_prec_macro, rel_rec_macro, rel_f1_macro,

                      rel_nec_prec_micro, rel_nec_rec_micro, rel_nec_f1_micro,
                      rel_nec_prec_macro, rel_nec_rec_macro, rel_nec_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        self._logger.info("Relations:")
        for r in input_reader.relation_types.values():
            self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro',
                                                 'rel_prec_macro', 'rel_rec_macro', 'rel_f1_macro',
                                                 'rel_nec_prec_micro', 'rel_nec_rec_micro', 'rel_nec_f1_micro',
                                                 'rel_nec_prec_macro', 'rel_nec_rec_macro', 'rel_nec_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})

    def _predict_unlabeled(self, model: torch.nn.Module, dataset: Dataset, input_reader: BaseInputReader, cnt: int):
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self._args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self._args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        pred_entities = []
        pred_relations = []

        with torch.no_grad():
            model.eval()

            # iterate batches
            doc_id = 0
            scores = []
            total = math.ceil(dataset.document_count / self._args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Predict'):
                # move batch to selected device
                batch = util.to_device(batch, self._device)
                # for k in batch:
                #     print("{}:{}".format(k, batch[k].shape))
                # print("------------------")

                # run model (forward pass)
                result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                               entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                               entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                               inference=True)
                entity_clf, rel_clf, rels = result
                # print(entity_clf[:5])
                # print(rel_clf[:5])

                # convert predictions
                predictions = prediction.convert_predictions_semi(entity_clf, rel_clf, rels,
                                                                  batch, self._args.semi_rel_filter_threshold,
                                                                  input_reader, ner_filter_threshold=self._args.semi_ner_filter_threshold)

                batch_pred_entities, batch_pred_relations, ave_score = predictions
                pred_entities.extend(batch_pred_entities)
                pred_relations.extend(batch_pred_relations)
                
                # for i in range(doc_id, doc_id+self._args.eval_batch_size):
                #     dataset._documents[i]._ent_score = ave_score[i-doc_id]
                #     scores.append(ave_score[i-doc_id])
                dataset._documents[doc_id]._ent_score = ave_score
                scores.append(ave_score)
                doc_id += self._args.eval_batch_size

        predictions = prediction.store_predictions_semi(dataset.documents, pred_entities, pred_relations, self._args.unlabeled_predictions_path)
        predictions = [item for item in zip(predictions, scores)]
        predictions.sort(key=lambda x:x[1], reverse=True)

        # filter low confidence instances
        strong_predictions = []
        for item in predictions[:cnt]:
            if item[1] < self._args.semi_ner_filter_threshold:
                break
            strong_predictions.append(item[0])
        predictions = strong_predictions

        with open(self._args.unlabeled_predictions_path, 'a+') as predictions_file:
            for item in predictions:
                print(json.dumps(item), file=predictions_file)

        top_cnt = dataset.remove(min(len(predictions),cnt))
        # filter empty instances
        available_predict = []
        for item in predictions:
            if len(item["entities"]) > 0 :
                available_predict.append(item)

        return available_predict
    

        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self._args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self._args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        pred_entities = []
        pred_relations = []

        with torch.no_grad():
            model.eval()

            # iterate batches
            doc_id = 0
            scores = []
            total = math.ceil(dataset.document_count / self._args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Predict'):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                               entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                               entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                               inference=True)
                entity_clf, rel_clf, rels = result
                # print("entity_clf:{}".format(entity_clf.shape))
                # print("rel_clf:{}".format(rel_clf.shape))
                # print("rels:{}".format(rels.shape))
                # print(entity_clf[:5])
                # print(rel_clf[:5])

                # convert predictions
                predictions = prediction.convert_predictions_semi(entity_clf, rel_clf, rels,
                                                                  batch, self._args.semi_rel_filter_threshold,
                                                                  input_reader, ner_filter_threshold=self._args.semi_ner_filter_threshold)

                batch_pred_entities, batch_pred_relations, ave_score = predictions
                pred_entities.extend(batch_pred_entities)
                pred_relations.extend(batch_pred_relations)
                
                # for i in range(doc_id, doc_id+self._args.eval_batch_size):
                #     dataset._documents[i]._ent_score = ave_score[i-doc_id]
                #     scores.append(ave_score[i-doc_id])
                dataset._documents[doc_id]._ent_score = ave_score
                scores.append(ave_score)
                doc_id += self._args.eval_batch_size

        predictions = prediction.store_predictions_semi(dataset.documents, pred_entities, pred_relations, self._args.unlabeled_predictions_path)
        for pred, score in zip(predictions, scores):
            pred["ner_score"] = score
        balancing.set_sample_weight_ner(predictions, ner_prob)
        balancing.set_sample_weight_rel(predictions, rel_prob)
        predictions = balancing.sample_data_by_rel(predictions)
        # predictions = balancing.sample_data_by_ner(predictions)
        cnt = min(cnt, len(predictions))
        # predictions.sort(key=lambda x:x["score"], reverse=True)
        predictions = predictions[:cnt]

        with open(self._args.unlabeled_predictions_path, 'a+') as predictions_file:
            for item in predictions:
                print(json.dumps(item), file=predictions_file)

        # filter empty instances
        available_predict = []
        for item in predictions:
            if len(item["entities"]) > 0 :
                available_predict.append(item)

        return available_predict

    def _predict_unlabeled_balancing(self, model: torch.nn.Module, dataset: Dataset, input_reader: BaseInputReader, cnt, ner_prob, rel_prob, unlabeled_type, sort_state):
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self._args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self._args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        pred_entities = []
        pred_relations = []

        with torch.no_grad():
            model.eval()

            # iterate batches
            doc_id = 0
            ner_scores = []
            rel_scores = []
            total = math.ceil(dataset.document_count / self._args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Predict'):
                # move batch to selected device
                batch = util.to_device(batch, self._device)
                # for k in batch:
                #     print("{}:{}".format(k, batch[k].shape))
                # print("------------------")

                # run model (forward pass)
                result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                               entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                               entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                               inference=True)
                entity_clf, rel_clf, rels = result
                # print(entity_clf[:5])
                # print(rel_clf[:5])

                # convert predictions
                predictions = prediction.convert_predictions_semi(entity_clf, rel_clf, rels,
                                                                  batch, self._args.semi_rel_filter_threshold,
                                                                  input_reader, ner_filter_threshold=self._args.semi_ner_filter_threshold)

                batch_pred_entities, batch_pred_relations, ave_ner_score, ave_rel_score = predictions
                pred_entities.extend(batch_pred_entities)
                pred_relations.extend(batch_pred_relations)
                
                # for i in range(doc_id, doc_id+self._args.eval_batch_size):
                #     dataset._documents[i]._ent_score = ave_score[i-doc_id]
                #     scores.append(ave_score[i-doc_id])
                dataset._documents[doc_id]._ent_score = ave_ner_score
                dataset._documents[doc_id]._rel_score = ave_rel_score
                ner_scores.append(ave_ner_score)
                rel_scores.append(ave_rel_score)
                doc_id += self._args.eval_batch_size

        predictions = prediction.store_predictions_semi(dataset.documents, pred_entities, pred_relations, self._args.unlabeled_predictions_path)
        for pred, ner_score, rel_score in zip(predictions, ner_scores, rel_scores):
            pred["ner_score"] = ner_score
            pred["rel_score"] = rel_score
        
        if sort_state:
            if unlabeled_type == "rel":
                predictions.sort(key=functools.cmp_to_key(util.cmp_rel), reverse=True)
            else:
                predictions.sort(key=functools.cmp_to_key(util.cmp_ner), reverse=True)

        balancing.set_sample_weight_ner(predictions, ner_prob)
        balancing.set_sample_weight_rel(predictions, rel_prob)
        if unlabeled_type == "rel":
            predictions = balancing.sample_data_by_rel(predictions)
        else:
            predictions = balancing.sample_data_by_ner(predictions)
        cnt = min(cnt, len(predictions))
        # predictions.sort(key=lambda x:x["score"], reverse=True)
        predictions = predictions[:cnt]

        with open(self._args.unlabeled_predictions_path, 'a+') as predictions_file:
            for item in predictions:
                print(json.dumps(item), file=predictions_file)

        # filter empty instances
        available_predict = []
        for item in predictions:
            if len(item["entities"]) > 0 :
                available_predict.append(item)

        return available_predict
    
    def _predict_unlabeled_balancing_ner_rel(self, model: torch.nn.Module, dataset: Dataset, input_reader: BaseInputReader, cnt, ner_prob, rel_prob, semi_cnt_control, sort_state):
        if semi_cnt_control:
            cnt = cnt // 2
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self._args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self._args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        pred_entities = []
        pred_relations = []

        with torch.no_grad():
            model.eval()

            # iterate batches
            doc_id = 0
            scores = []
            total = math.ceil(dataset.document_count / self._args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Predict'):
                # move batch to selected device
                batch = util.to_device(batch, self._device)
                # for k in batch:
                #     print("{}:{}".format(k, batch[k].shape))
                # print("------------------")

                # run model (forward pass)
                result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                               entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                               entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                               inference=True)
                entity_clf, rel_clf, rels = result
                # print(entity_clf[:5])
                # print(rel_clf[:5])

                # convert predictions
                predictions = prediction.convert_predictions_semi(entity_clf, rel_clf, rels,
                                                                  batch, self._args.semi_rel_filter_threshold,
                                                                  input_reader, ner_filter_threshold=self._args.semi_ner_filter_threshold)

                batch_pred_entities, batch_pred_relations, ave_score = predictions
                pred_entities.extend(batch_pred_entities)
                pred_relations.extend(batch_pred_relations)
                
                # for i in range(doc_id, doc_id+self._args.eval_batch_size):
                #     dataset._documents[i]._ent_score = ave_score[i-doc_id]
                #     scores.append(ave_score[i-doc_id])
                dataset._documents[doc_id]._ent_score = ave_score
                scores.append(ave_score)
                doc_id += self._args.eval_batch_size

        predictions = prediction.store_predictions_semi(dataset.documents, pred_entities, pred_relations, self._args.unlabeled_predictions_path)
        for pred, score in zip(predictions, scores):
            pred["ner_score"] = score
        balancing.set_sample_weight_ner(predictions, ner_prob)
        balancing.set_sample_weight_rel(predictions, rel_prob)
        predictions_rel = balancing.sample_data_by_rel(predictions)
        predictions_ner = balancing.sample_data_by_ner(predictions)
        cnt_ner = min(cnt, len(predictions_ner))
        cnt_rel = min(cnt, len(predictions_rel))
        # predictions.sort(key=lambda x:x["score"], reverse=True)
        predictions_ner = predictions_ner[:cnt_ner]
        predictions_rel = predictions_rel[:cnt_rel]
        predictions = predictions_ner + predictions_rel

        with open(self._args.unlabeled_predictions_path, 'a+') as predictions_file:
            for item in predictions:
                print(json.dumps(item), file=predictions_file)

        # filter empty instances
        available_predict = []
        for item in predictions:
            if len(item["entities"]) > 0 :
                available_predict.append(item)

        return available_predict