# Installation

Tested with **Python 3.10, PyTorch 2.5.1, CUDA 11.8, spconv-cu118**.

### Requirements

* Python 3.10
* PyTorch 2.5.1
* CUDA 11.8
* spconv v2.x

### Steps

```bash
# 1. create the environment
conda create -n autoreg3d python=3.10
conda activate autoreg3d

# 2. install PyTorch (CUDA 11.8) and spconv
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu118

pip install spconv-cu118

# 3. install the remaining dependencies
pip install -r requirements.txt

# 4. build and install pcdet
python setup.py develop
```

### Mamba-based backbone (optional)

Required only for the Mamba-based LION backbone
(`LION3DBackboneOneStride`). Operator imports are lazy, so the base
install does not depend on this.

```bash
pip install causal-conv1d==1.5.0.post8 torch_scatter timm einops
cd pcdet/ops/mamba && python setup.py install && cd -
```

The other LION operators (RWKV / RetNet / xLSTM / TTT) are not bundled
with this release. Please refer to [LION](https://github.com/happinesslz/LION) for setup instructions.