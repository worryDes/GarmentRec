# GarmentRec

Official implementation of **GarmentRec (IEEE TIP 2026): Individual Garment Reconstruction from a Monocular Human Image**.

## Environment

Create the conda environment using:

```bash
conda env create -f environment.yml
conda activate GarmentRec
```

## Dataset

The dataset introduced in our paper can be downloaded from:

[Dataset Link]

After downloading, organize the dataset according to the structure described in the paper.

## Pretrained Models

Download the pretrained model and place it under:

```text
./models/mrf_0.1_shading_0.1/
```

## Inference

Place input images in:

```text
./test_images
```

Run:

```bash
python code/infer.py \
    --model_path ./models/mrf_0.1_shading_0.1/mrf_0.1_shading_0.1_pca64_ep100_bth0.pth \
    --displacement_scale 0.005 \
    --input_folder ./test_images \
    --output_folder ./Results/test_images
```

### Normal Refinement (Optional)

For normal refinement, additional dependencies are required. Please install them before running inference.

Enable normal refinement by adding:

```bash
--normal_refine 1
```

In this mode, each input image should be accompanied by:

* Upper-garment mask:

  ```text
  xxx_mask_up.png
  ```

* Lower-garment mask:

  ```text
  xxx_mask_bottom.png
  ```

* Surface normal map:

  ```text
  xxx_normal.png
  ```

Example:

```text
0001.png
0001_mask_up.png
0001_mask_bottom.png
0001_normal.png
```

## Output

The reconstructed garment meshes and visualization results will be saved to:

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
