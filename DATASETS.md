# How to install datasets

Please create a folder named `datasets/` and download the following datasets under it.
- [Caltech101](#Caltech101)
- [CUB](#CUB)
- [Flowers102](#Flowers102)
- [Food101](#Food101)
- [OxfordIIITPet](#OxfordIIITPet)
- [StanfordDogs](#StanfordDogs)
- [BronzeDing](#BronzeDing)

The instructions to prepare each dataset are detailed below.

### Caltech101
- Create a folder named `Caltech/` under `datasets/`.
- Download `caltech-101.zip` from [link](https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1) and extract it under `datasets/Caltech/`.
- Download `split_zhou_Caltech101.json` from [link](https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view?usp=sharing) and put it under `datasets/Caltech/caltech-101/` (this split is from [CoOp]). (https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1)

The directory structure should look like
```
--datasets/
    |-- Caltech/
        |-- caltech-101/
            |–– 101_ObjectCategories/
            |–– split_zhou_Caltech101.json
```

### CUB
- Create a folder named `CUB/` under `datasets/`.
- Download `CUB_200_2011.tgz` from [link](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1) and extract it under `datasets/CUB/`.

The directory structure should look like
```
--datasets/
    |-- CUB/
        |-- CUB_200_2011/
            |–– images/
            |–– images.txt
            |–– train_test_split.txt
            |-- image_class_labels.txt
```

### Flowers102
- Create a folder named `Flowers102/` under `datasets/`.
- Download `102flowers.tgz` from [link](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) and extract it under `datasets/Flowers102/`.
- Download `imagelabels.mat` from [link](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat) and put it under `datasets/Flowers102/flowers-102/`. 
- Download `setid.mat` from [link](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat) and put it under `datasets/Flowers102/flowers-102/`. 

The directory structure should look like
```
--datasets/
    |-- Flowers102/
        |-- flowers-102/
            |–– jpg/
            |-- imagelabels.mat
            |-- setid.mat
```

### Food101
- Create a folder named `Food101/` under `datasets/`.
- Download `food-101.tar.gz` from [link](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz) and extract it under `datasets/Food101/`.

The directory structure should look like
```
--datasets/
    |-- Food101/
        |-- food-101/
            |–– images/
            |-- meta/
```

### OxfordIIITPet
- Create a folder named `OxfordIIITPet/` under `datasets/`.
- Download `images.tar.gz` from [link](https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz) and extract it under `datasets/OxfordIIITPet/`.
- Download `annotations.tar.gz` from [link](https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz) and extract it under `datasets/OxfordIIITPet/`. 

The directory structure should look like
```
--datasets/
    |-- OxfordIIITPet/
        |-- oxford-iiit-pet/
            |–– images/
            |-- annotations/
```

### StanfordDogs
- Create a folder named `StanfordDogs/` under `datasets/`.
- Download `images.tar` from [link](http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar) and extract it under `datasets/StanfordDogs/`.
- Download `annotation.tar` from [link](http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar) and extract it under `datasets/StanfordDogs/`.
- Download `lists.tar` from [link](http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar) and extract it under `datasets/StanfordDogs/`.

The directory structure should look like
```
--datasets/
    |-- StanfordDogs/
        |-- Images/
        |-- Annotation/
        |-- file_list.mat
        |-- train_list.mat
        |-- test_list.mat
```

### BronzeDing
- Create a folder named `BronzeDing/` under `datasets/`.
- Download `DATASET.zip` from [link](https://github.com/zhourixin/bronze-Ding) and extract it,
then put `complete_DATASET/` under `datasets/BronzeDing/`.

The directory structure should look like
```
--datasets/
    |-- BronzeDing/
        |-- complete_DATASET/
            |-- delete_png/
            |-- xml_all/
            |–– train.xlsx
            |-- test.xlsx
```
