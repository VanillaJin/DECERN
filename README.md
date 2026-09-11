# Official Implementation of ECCV 2026 paper "Combining Discrepancy-Confusion Uncertainty and Calibration Diversity for Active Fine-Grained Image Classification"

## Get Start

### Environmental Dependencies
- python 3.9
- torch 1.13.1+cu117
- torchvision 0.14.1+cu117

### Datasets
We use the following fine-grained image classification dataset
- [Caltech101](https://data.caltech.edu/records/mzrjq-6wc02)
- [CUB](https://drive.google.com/drive/folders/1kFzIqZL_pEBVR7Ca_8IKibfWoeZc3GT1)
- [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [OxfordIIITPet](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [StanfordDogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [BronzeDing](https://github.com/zhourixin/bronze-Ding)

More details are available at [DATASETS.md](DATASETS.md)

## How to Run
```bash
CUDA_VISIBLE_DEVICES=${gpu_ids} python main.py \
--deterministic \
-d ${data} \
-m ${model} \
--pretrained \
-al ${method} \
--init_budget_x ${init_budget_x} \
--comm_budget_x ${comm_budget_x}
```
- `${gpu_ids}`: GPU ids, e.g. `0`
- `${data}`: Dataset name, e.g. `Caltech101`, `CUB`, `Food101`, `Flowers102`, `OxfordIIITPet`, `StanfordDogs`, `BronzeDing`
- `${model}`: Model name, e.g. `ResNet50`, `ViT_Small`, `DINOv2`
- `${method}`: Active learning method, e.g. `random`, `decern`
- `${init_budget_x}`: Initial budget, e.g. `1`, `2`
- `${comm_budget_x}`: Common budget, e.g. `1`, `2`

### Example
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--deterministic \
-d BronzeDing \
-m ResNet50 \
--pretrained \
-al decern \
--init_budget_x 1 \
--comm_budget_x 1
```

## Citation

```bibtex
@misc{jin2025combiningdiscrepancyconfusionuncertaintycalibration,
      title={Combining Discrepancy-Confusion Uncertainty and Calibration Diversity for Active Fine-Grained Image Classification}, 
      author={Yinghao Jin and Xi Yang},
      year={2025},
      eprint={2509.24181},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.24181}, 
}
```

## Acknowledgements
