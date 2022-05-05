CUDA_VISIBLE_DEVICES=$1 python semi_spert.py train \
  --config configs/tune_ner_th.conf \
  --label ade2scierc_wo_generic_tune_10-$2-$3 \
  --train_path data/datasets/scierc_wo_generic/labeled-10-$2.json \
  --semi_ner_filter_threshold 0.$3