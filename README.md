## Usage

### 1. Prepare data

Please go to ["./datasets/README.md"](datasets/README.md) for details

### 2. Environment

Please prepare an environment with python=3.9, and then use the command "pip install -r requirements.txt" for the dependencies.

### 3. Train/Test

- Run the train script on synapse dataset. The batch size can be reduced to 12 or 6 to save memory (please also decrease the base_lr linearly), and both can reach similar performance.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
```

- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
```
