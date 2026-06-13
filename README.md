# GarmentRec

Official implementation of **GarmentRec (IEEE TIP 2026): Individual Garment Reconstruction from a Monocular Human Image**.

## Environment

```bash id="q1m8zv"
conda env create -f environment.yml
conda activate GarmentRec
```

## Dataset

The dataset introduced in our paper can be downloaded from:

[Dataset Link]

## Preparation

Download the required assets from:

* [Google Drive](https://drive.google.com/file/d/1PTbfEMchwgHpaL3y8Gm1__sbzFLqfoYj/view?usp=drive_link)

and extract them under `data/`.

Please follow the instructions in `smpl_pytorch/README.md` to download SMPL models and put them under `smpl_pytorch/model/`.

## Pretrained Models

[Coming Soon]

## Inference

```bash id="n4k9xa"
python code/infer.py \
    --model_path ./models/mrf_0.1_shading_0.1/mrf_0.1_shading_0.1_pca64_ep100_bth0.pth \
    --displacement_scale 0.005 \
    --input_folder ./test_images \
    --output_folder ./Results/test_images
```

For normal refinement, add:

```bash id="z8v3cq"
--normal_refine 1
```

and provide:

```text id="t2k6pd"
xxx.png
xxx_mask_up.png
xxx_mask_bottom.png
xxx_normal.png
```

## Output

Results will be saved to:

```text id="f7m2lw"
./Results/test_images
```

## Citation

```bibtex id="c9r1hx"
@article{GarmentRec2026,
  title={Individual Garment Reconstruction from a Monocular Human Image},
  author={...},
  journal={IEEE Transactions on Image Processing},
  year={2026}
}
```
