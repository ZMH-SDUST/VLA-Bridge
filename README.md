# Bridging Feature Misalignment and Semantic Confusion for Zero-Shot HOI Detection

## Dataset
Download the official HICO-DET and V-COCO datasets. The download files should organized as follows:
```
|- VLA-Bridge  
|   |- hicodet  
|   |   |- hico_20160224_det  
|   |       |- annotations  
|   |       |- images  
|   |- vcoco  
|   |   |- mscoco2014  
|   |       |- train2014  
|   |       |- val2014      
```

## Dependencies
1. Follow the environment setup in [UPT](https://github.com/fredzzhang/upt).
2. Install the local package of CLIP.
3. Download the CLIP weights [ViT-B-16.pt, ViT-L-14-336px.pt] to `VLA-Bridge/checkpoints/`.
4. Download the DETR weights [detr-r50-hicodet.pth, detr-r50-vcoco.pth] to `VLA-Bridge/checkpoints/`.
5. Download the pre-extracted features from [ADA-CM](https://github.com/ltttpku/ADA-CM).
