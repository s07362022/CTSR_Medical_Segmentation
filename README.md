# CTSR
Self-Training Methodology for Enhancing
Semi-Supervised Semantic Segmentation Through
Selection of Reliable Pseudo Labels:Application on
Ki67 Markers in Breast Cancer Tissue

## Lab :
Smile Lab from NCKU in Taiwan

## Getting Started

```sh
sh sup_start.sh
```

## Training and Testing
'''
export semi_setting='ki67/ex25'

CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore main_medical_DFCA_a.py \
  --dataset ki67 --data-root  [Your Pascal Path] \
  --batch-size 8 --backbone resnet50 --model unetpp \
  --labeled-id-path dataset/splits/$semi_setting/labeled.txt \
  --unlabeled-id-path dataset/splits/$semi_setting/unlabeled.txt \
  --pseudo-mask-path outdir/pseudo_masks_ki67/$semi_setting \
  --reliable-id-path /home/u5914116/Harden/SOTA_papper_code/ST-mysel/outdir/reliable_ids_ki67 \
  --save-path outdir/models/$semi_setting   
'''
