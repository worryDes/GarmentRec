## GarmentRec

Official implementation of **GarmentRec (IEEE TIP 2026): Individual Garment Reconstruction from a Monocular Human Image**

📄 **Paper:** [Coming Soon]

---

## ⚙️ Environment

```bash id="env2"
conda env create -f environment.yml
conda activate GarmentRec
```

---

## 📦 Dataset

The dataset introduced in our paper will be released soon.

👉 [Dataset Link]

---

## 🛠️ Preparation

Download required assets:

* 📥 [Google Drive](https://drive.google.com/file/d/1PTbfEMchwgHpaL3y8Gm1__sbzFLqfoYj/view?usp=drive_link)

Place them under:

```text id="prep1"
data/
```

For SMPL models, please follow instructions in `smpl_pytorch/README.md` and place them under:

```text id="prep2"
smpl_pytorch/model/
```

---

## 🧠 Pretrained Models

Download pretrained weights:

* 📥 [Google Drive](https://drive.google.com/file/d/1EF_BnrsZ0Zk53IVLCaa00xSmLdaPXFcF/view?usp=sharing)

Place them under:

```text id="model1"
models/
```

---

## 🚀 Inference

### Standard inference

```bash id="inf2"
python code/infer.py \
    --model_path ./models/mrf_0.1_shading_0.1/mrf_0.1_shading_0.1_pca64_ep100_bth0.pth \
    --displacement_scale 0.005 \
    --input_folder ./test_images \
    --output_folder ./Results/test_images
```

---

### ✨ Normal Refinement (optional)

Add:

```bash id="nr3"
--normal_refine 1
```

Input per image:

```text id="nr4"
xxx.png
xxx_mask_up.png
xxx_mask_bottom.png
xxx_normal.png
```

---

## 📤 Output

Results will be saved to:

```text id="out2"
Results/test_images/
```

---

## 📚 Citation

```bibtex id="bib2"
@article{GarmentRec2026,
  title={Individual Garment Reconstruction from a Monocular Human Image},
  author={...},
  journal={IEEE Transactions on Image Processing},
  year={2026}
}
```
