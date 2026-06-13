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

Download the required assets from:

* [Google Drive](https://drive.google.com/file/d/1PTbfEMchwgHpaL3y8Gm1__sbzFLqfoYj/view?usp=drive_link)

Extract the downloaded archive and place the contents under:

```text
./data
```

## Pretrained Models

[Coming Soon]

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

and provide:

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
