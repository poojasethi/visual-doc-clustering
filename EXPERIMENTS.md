Commands to run to reproduce experiment results.

# RVL-CDIP

## Bag of Words
```
python clustering.py -p datasets/rvl-cdip/ -r rivlet_count -e 768 -o results/rvl-cdip/rivlet_count/
```

## TF-IDF
```
python clustering.py -p datasets/rvl-cdip/ -r rivlet_tfidf -e 768 -o results/rvl-cdip/rivlet_tfidf/
```

## Vanilla LayoutLMV1

### Average all words
```
python clustering.py -p datasets/rvl-cdip/ -r vanilla_lmv1 -e 768 -s average_all_words -o results/rvl-cdip/vanilla_lmv1/average_all_words
```

### Average all words, mask pads, append sequence length
```
python clustering.py -p datasets/rvl-cdip/ -r vanilla_lmv1 -e 769 -s average_all_words_mask_pads -o results/rvl-cdip/vanilla_lmv1/average_all_words_mask_pads
```

### Average all words, mask pads, append normalized sequence length
```
python clustering.py -p datasets/rvl-cdip/ -r vanilla_lmv1 -e 769 -s average_all_words_mask_pads -o results/rvl-cdip/vanilla_lmv1/average_all_words_mask_pads_normalized -n
```

### Last word, append sequence length
```
python clustering.py -p datasets/rvl-cdip/ -r vanilla_lmv1 -e 769 -s last_word -o results/rvl-cdip/vanilla_lmv1/last_word
```

## Finetuned LayoutLMV1 (Related)

### Average all words
```
python clustering.py -p datasets/rvl-cdip/ -r finetuned_related_lmv1 -e 768 -s average_all_words -o results/rvl-cdip/finetuned_related_lmv1/average_all_words
```

### Average all words, mask pads, append sequence length
```
python clustering.py -p datasets/rvl-cdip/ -r finetuned_related_lmv1 -e 769 -s average_all_words_mask_pads -o results/rvl-cdip/finetuned_related_lmv1/average_all_words_mask_pads
```

### Average all words, mask pads, append normalized sequence length
```
python clustering.py -p datasets/rvl-cdip/ -r finetuned_related_lmv1 -e 769 -s average_all_words_mask_pads -o results/rvl-cdip/finetuned_related_lmv1/average_all_words_mask_pads_normalized -n
```

### Last word, append sequence length
```
python clustering.py -p datasets/rvl-cdip/ -r finetuned_related_lmv1 -e 769 -s last_word -o results/rvl-cdip/finetuned_related_lmv1/last_word
```

## Finetuned LayoutLMV1 (Unrelated)

### Average all words
```
python clustering.py -p datasets/rvl-cdip/ -r finetuned_unrelated_lmv1 -e 768 -s average_all_words -o results/rvl-cdip/finetuned_unrelated_lmv1/average_all_words
```

### Average all words, mask pads, append sequence length
```
python clustering.py -p datasets/rvl-cdip/ -r finetuned_unrelated_lmv1 -e 769 -s average_all_words_mask_pads -o results/rvl-cdip/finetuned_unrelated_lmv1/average_all_words_mask_pads
```

### Average all words, mask pads, append normalized sequence length
```
python clustering.py -p datasets/rvl-cdip/ -r finetuned_unrelated_lmv1 -e 769 -s average_all_words_mask_pads -o results/rvl-cdip/finetuned_unrelated_lmv1/average_all_words_mask_pads_normalized -n
```

### Last word, append sequence length
```
python clustering.py -p datasets/rvl-cdip/ -r finetuned_unrelated_lmv1 -e 769 -s last_word -o results/rvl-cdip/finetuned_unrelated_lmv1/last_word
```

##  Vanilla LayoutLMV2

### Average all words
```
python clustering.py -p datasets/rvl-cdip/ -r vanilla_lmv2 -e 768 -s average_all_words -o results/rvl-cdip/vanilla_lmv2/average_all_words
```

### Average all words, mask pads, append sequence length
```
python clustering.py -p datasets/rvl-cdip/ -r vanilla_lmv2 -e 769 -s average_all_words_mask_pads -o results/rvl-cdip/vanilla_lmv2/average_all_words_mask_pads
```

### Average all words, mask pads, append normalized sequence length
```
python clustering.py -p datasets/rvl-cdip/ -r vanilla_lmv2 -e 769 -s average_all_words_mask_pads -o results/rvl-cdip/vanilla_lmv2/average_all_words_mask_pads_normalized -n
```

### Last word, append sequence length
```
python clustering.py -p datasets/rvl-cdip/ -r vanilla_lmv2 -e 769 -s last_word -o results/rvl-cdip/vanilla_lmv2/last_word
```

================

# SROIE 2019

## Bag of Words
```
python clustering.py -p datasets/sroie2019/ -r rivlet_count -e 768 -o results/sroie2019/rivlet_count/
```

## TF-IDF
```
python clustering.py -p datasets/sroie2019/ -r rivlet_tfidf -e 768 -o results/sroie2019/rivlet_tfidf/
```

## Vanilla LayoutLMV1

### Average all words, mask pads, append sequence length
```
python clustering.py -p datasets/sroie2019/ -r vanilla_lmv1 -e 768 -s average_all_words -o results/sroie2019/vanilla_lmv1/average_all_words
```

### Average all words, mask pads, append normalized sequence length
```
python clustering.py -p datasets/sroie2019/ -r vanilla_lmv1 -e 768 -s average_all_words -o results/sroie2019/vanilla_lmv1/average_all_words_normalized -n
```

### Average all words mask pads
```
python clustering.py -p datasets/sroie2019/ -r vanilla_lmv1 -e 769 -s average_all_words_mask_pads -o results/sroie2019/vanilla_lmv1/average_all_words_mask_pads
```

### Last word, append sequence length
```
python clustering.py -p datasets/sroie2019/ -r vanilla_lmv1 -e 769 -s last_word -o results/sroie2019/vanilla_lmv1/last_word
```

## Finetuned LayoutLMV1 (Related)

### Average all words
```
python clustering.py -p datasets/sroie2019/ -r finetuned_related_lmv1 -e 768 -s average_all_words -o results/sroie2019/finetuned_related_lmv1/average_all_words
```

### Average all words, mask pads, append sequence length
```
python clustering.py -p datasets/sroie2019/ -r finetuned_related_lmv1 -e 769 -s average_all_words_mask_pads -o results/sroie2019/finetuned_related_lmv1/average_all_words_mask_pads
```

### Average all words, mask pads, append normalized sequence length
```
python clustering.py -p datasets/sroie2019/ -r finetuned_related_lmv1 -e 769 -s average_all_words_mask_pads -o results/sroie2019/finetuned_related_lmv1/average_all_words_mask_pads_normalized -n
```

### Last word, append sequence length
```
python clustering.py -p datasets/sroie2019/ -r finetuned_related_lmv1 -e 769 -s last_word -o results/sroie2019/finetuned_related_lmv1/last_word
```

## Finetuned LayoutLMV1 (Unrelated)

### Average all words
```
python clustering.py -p datasets/sroie2019/ -r finetuned_unrelated_lmv1 -e 768 -s average_all_words -o results/sroie2019/finetuned_unrelated_lmv1/average_all_words
```

### Average all words mask pads
```
python clustering.py -p datasets/sroie2019/ -r finetuned_unrelated_lmv1 -e 769 -s average_all_words_mask_pads -o results/sroie2019/finetuned_unrelated_lmv1/average_all_words_mask_pads
```

### Average all words, mask pads, append sequence length
```
python clustering.py -p datasets/sroie2019/ -r finetuned_related_lmv1 -e 769 -s average_all_words_mask_pads -o results/sroie2019/finetuned_related_lmv1/average_all_words_mask_pads
```

### Average all words, mask pads, append normalized sequence length
```
python clustering.py -p datasets/sroie2019/ -r finetuned_related_lmv1 -e 769 -s average_all_words_mask_pads -o results/sroie2019/finetuned_related_lmv1/average_all_words_mask_pads_normalized -n
```

### Last word, append sequence length
```
python clustering.py -p datasets/sroie2019/ -r finetuned_unrelated_lmv1 -e 769 -s last_word -o results/sroie2019/finetuned_unrelated_lmv1/last_word
```

##  Vanilla LayoutLMV2

### Average all words
```
python clustering.py -p datasets/sroie2019/ -r vanilla_lmv2 -e 768 -s average_all_words -o results/sroie2019/vanilla_lmv2/average_all_words
```

### Average all words, mask pads, append sequence length
```
python clustering.py -p datasets/sroie2019/ -r vanilla_lmv2 -e 769 -s average_all_words_mask_pads -o results/sroie2019/vanilla_lmv2/average_all_words_mask_pads
```

### Average all words, mask pads, append normalized sequence length
```
python clustering.py -p datasets/sroie2019/ -r vanilla_lmv2 -e 769 -s average_all_words_mask_pads -o results/sroie2019/vanilla_lmv2/average_all_words_mask_pads_normalized -n
```

### Last word, append sequence length
```
python clustering.py -p datasets/sroie2019/ -r vanilla_lmv2 -e 769 -s last_word -o results/sroie2019/vanilla_lmv2/last_word
```
