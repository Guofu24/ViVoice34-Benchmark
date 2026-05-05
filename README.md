# Experimental Code and Reproducibility Guide

This repository contains the implementation and experimental scripts used in our study.  
The code is organized by experiment type so that each result can be reproduced independently.

## 1. Repository Structure

```text
.
├───ASR
│       evaluate_zipformer_ViVoice34.ipynb
│       
├───Data Preprocessing
│       create_similarity_excel.py
│       create_trial_val.py
│       delete_speaker_same.py
│       filter_speakers_by_duration.py
│       merge_gallery_query.py
│       model_arch.py
│       pipeline.py
│       
├───Demographic Classification
│   ├───Age Group CLS
│   │       wav2vec-age-classification-01.ipynb
│   │       wav2vec-age-classification-02.ipynb
│   │       
│   ├───Dialect CLS
│   │       wavvec-dialect.ipynb
│   │       whisper_dialect.ipynb
│   │       
│   └───Gender CLS
│           vimd-gender-classification-clean.ipynb
│           
└───Speaker Verification
    ├───ECAPA-TDNN
    │       evaluate-new-ecapa.ipynb
    │       training_ecapatdnn.ipynb
    │       
    ├───REDIMNET
    │       evaluate-redimnet-13-03-2026.ipynb
    │       training_ReDimNet.ipynb
    │       
    └───SAMRESNET
            samres34-15-04.ipynb
            training_Samresnet34.ipynb