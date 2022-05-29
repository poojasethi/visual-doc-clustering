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


