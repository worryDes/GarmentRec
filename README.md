# GarmentRec

Official implementation of **GarmentRec (IEEE TIP 2026): Individual Garment Reconstruction from a Monocular Human Image**.

## Environment

```bash
conda env create -f environment.yml
conda activate GarmentRec
```

## Dataset

The dataset introduced in our paper can be downloaded from:

[Dataset Link]

## Preparation

Before inference, please download:

* Pretrained model weights
* PCA garment templates
* Other required assets

and place them in the corresponding folders.

[Download Link]

## Inference

```bash
python code/infer.py \
    --model_path ./models/mrf_0.1_shading_0.1/mrf_0.1_shading_0.1_pca64_ep100_bth0.pth \
    --displacement_scale 0.005 \
    --input_folder ./test_images \
    --output_folder ./Results/test_images
```

For normal refinement, add:

```bash
--normal_refine 1
```

and provide the following files for each image:

```text
xxx.png
xxx_mask_up.png
xxx_mask_bottom.png
xxx_normal.png
```

## Output

Results will be saved to:

```text
./Results/test_images
```

## Citation

```bibtex
@article{GarmentRec2026,
  title={Individual Garment Reconstruction from a Monocular Human Image},
  author={...},
  journal={IEEE Transactions on Image Processing},
  year={2026}
}
```
