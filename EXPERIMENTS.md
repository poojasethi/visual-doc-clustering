Commands to run to reproduce experiment results.

# SROIE2019

## ResNet
```
mkdir -p results/sroie2019/resnet/
python clustering.py -p datasets/sroie2019/ \
	-r resnet \
	-o results/sroie2019/resnet/ \
	--debug
```

## AlexNet
```
mkdir -p results/sroie2019/alexnet/
python clustering.py -p datasets/sroie2019/ \
	-r alexnet \
	-o results/sroie2019/alexnet/ \
	--debug
```

## LayoutLM Base ([CLS] Token)
```
mkdir -p results/sroie2019/layoutlm_base/cls_token/
python clustering.py -p datasets/sroie2019/ \
	-r layoutlm_base \
	-s cls_token \
	-o results/sroie2019/layoutlm_base/cls_token/ \
	--debug
```

## LayoutLM Base ([SEP] Token)
```
mkdir -p results/sroie2019/layoutlm_base/sep_token/
python clustering.py -p datasets/sroie2019/ \
	-r layoutlm_base \
	-s sep_token \
	-o results/sroie2019/layoutlm_base/sep_token/ \
	--debug
```

## LayoutLM Base (Average Tokens)
```
mkdir -p results/sroie2019/layoutlm_base/average_all_tokens/
python clustering.py -p datasets/sroie2019/ \
	-r layoutlm_base \
	-s average_all_tokens \
	-o results/sroie2019/layoutlm_base/average_all_tokens/ \
	--debug
```

## LayoutLM Large ([SEP] Token)
```
mkdir -p results/sroie2019/layoutlm_large/sep_token/
python clustering.py -p datasets/sroie2019/ \
	-r layoutlm_large \
	-s sep_token \
	-o results/sroie2019/layoutlm_large/sep_token/ \
	--debug
```

## LayoutLMv2 Base ([CLS] Token)
```
mkdir -p results/sroie2019/layoutlm_v2_base/cls_token/
python clustering.py -p datasets/sroie2019/ \
	-r layoutlm_v2_base \
	-s cls_token \
	-o results/sroie2019/layoutlm_v2_base/cls_token/ \
	--debug
```
## LayoutLMv2 Base ([SEP] Token)
```
mkdir -p results/sroie2019/layoutlm_v2_base/sep_token/
python clustering.py -p datasets/sroie2019/ \
	-r layoutlm_v2_base \
	-s sep_token \
	-o results/sroie2019/layoutlm_v2_base/sep_token/ \
	--debug
```

## LayoutLMv2 Base (Image Token)
```
mkdir -p results/sroie2019/layoutlm_v2_base/image_tokens/
python clustering.py -p datasets/sroie2019/ \
	-r layoutlm_v2_base \
	-s image_tokens \
	-o results/sroie2019/layoutlm_v2_base/image_tokens/ \
	--debug
```

## LayoutLMv2 Base (Average Tokens)
```
mkdir -p results/sroie2019/layoutlm_v2_base/average_all_tokens/
python clustering.py -p datasets/sroie2019/ \
	-r layoutlm_v2_base \
	-s average_all_tokens \
	-o results/sroie2019/layoutlm_v2_base/average_all_tokens/ \
	--debug
```

## LayoutLMv2 Large (Image Tokens)
```
mkdir -p results/sroie2019/layoutlm_v2_large/image_tokens/
python clustering.py -p datasets/sroie2019/ \
	-r layoutlm_v2_large \
	-s image_tokens \
	-o results/sroie2019/layoutlm_v2_large/image_tokens/ \
	--debug
```

# RVL-CDIP

## ResNet
```
mkdir -p results/rvl-cdip/resnet/
python clustering.py -p datasets/rvl-cdip/ \
	-r resnet \
	-o results/rvl-cdip/resnet/ \
	--debug
```

## AlexNet
```
mkdir -p results/rvl-cdip/alexnet/
python clustering.py -p datasets/rvl-cdip/ \
	-r alexnet \
	-o results/rvl-cdip/alexnet/ \
	--debug
```

## LayoutLM Base ([CLS] Token)
```
mkdir -p results/rvl-cdip/layoutlm_base/cls_token/
python clustering.py -p datasets/rvl-cdip/ \
	-r layoutlm_base \
	-s cls_token \
	-o results/rvl-cdip/layoutlm_base/cls_token/ \
	--debug
```

## LayoutLM Base ([SEP] Token)
```
mkdir -p results/rvl-cdip/layoutlm_base/sep_token/
python clustering.py -p datasets/rvl-cdip/ \
	-r layoutlm_base \
	-s sep_token \
	-o results/rvl-cdip/layoutlm_base/sep_token/ \
	--debug
```

## LayoutLM Base (Average Tokens)
```
mkdir -p results/rvl-cdip/layoutlm_base/average_all_tokens/
python clustering.py -p datasets/rvl-cdip/ \
	-r layoutlm_base \
	-s average_all_tokens \
	-o results/rvl-cdip/layoutlm_base/average_all_tokens/ \
	--debug
```
## LayoutLM Large ([SEP] Token)
```
mkdir -p results/rvl-cdip/layoutlm_large/sep_token/
python clustering.py -p datasets/rvl-cdip/ \
	-r layoutlm_large \
	-s sep_token \
	-o results/rvl-cdip/layoutlm_large/sep_token/ \
	--debug
```
## LayoutLMv2 Base ([CLS] Token)
```
mkdir -p results/rvl-cdip/layoutlm_v2_base/cls_token/
python clustering.py -p datasets/rvl-cdip/ \
	-r layoutlm_v2_base \
	-s cls_token \
	-o results/rvl-cdip/layoutlm_v2_base/cls_token/ \
	--debug
```

## LayoutLMv2 Base ([SEP] Token)
```
mkdir -p results/rvl-cdip/layoutlm_v2_base/sep_token/
python clustering.py -p datasets/rvl-cdip/ \
	-r layoutlm_v2_base \
	-s sep_token \
	-o results/rvl-cdip/layoutlm_v2_base/sep_token/ \
	--debug
```

## LayoutLMv2 Base (Image Token)
```
mkdir -p results/rvl-cdip/layoutlm_v2_base/image_tokens/
python clustering.py -p datasets/rvl-cdip/ \
	-r layoutlm_v2_base \
	-s image_tokens \
	-o results/rvl-cdip/layoutlm_v2_base/image_tokens/ \
	--debug
```

## LayoutLMv2 Base (Average Tokens)
```
mkdir -p results/rvl-cdip/layoutlm_v2_base/average_all_tokens/
python clustering.py -p datasets/rvl-cdip/ \
	-r layoutlm_v2_base \
	-s average_all_tokens \
	-o results/rvl-cdip/layoutlm_v2_base/average_all_tokens/ \
	--debug
```

## LayoutLMv2 Large (Image Tokens)
```
mkdir -p results/rvl-cdip/layoutlm_v2_large/image_tokens/
python clustering.py -p datasets/rvl-cdip/ \
	-r layoutlm_v2_large \
	-s image_tokens \
	-o results/rvl-cdip/layoutlm_v2_large/image_tokens/ \
	--debug
```

# ML Papers
```
mkdir -p results/ml-papers/layoutlm_base/sep_token/
python clustering.py -p datasets/ml-papers \
	-k 3 \
	-r layoutlm_base \
	-s sep_token \
	-o results/ml-papers/layoutlm_base/sep_token \
	--debug
```
```
mkdir -p results/ml-papers/layoutlm_v2_base/image_tokens/
python clustering.py -p datasets/ml-papers \
	-k 3 \
	-r layoutlm_v2_base \
	-s sep_token \
	-o results/ml-papers/layoutlm_v2_base/image_tokens \
	--debug
```

