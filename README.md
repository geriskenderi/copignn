## Combinatorial optimization with physics-inspired graph neural networks

This repository is an ongoing implementation of [Combinatorial optimization with physics-inspired graph neural networks](https://www.nature.com/articles/s42256-022-00468-6) done with PyTorch Lightning and PyTorch Geometric.

The current version implements only the Maximum Cut and Maximum Independent Set problems on random d-regular graphs, as explained in the paper. Please beware that the results are still inconclusive and this is an ongoing implementation, your feedback is more than welcomed.


### Python environment setup with Conda

```
conda create --name copignn python=3.9
conda activate copignn
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install pytorch-lightning -c conda-forge
conda install -c pyg pytorch-sparse
conda install -c pyg pytorch-scatter
conda install -c pyg pytorch-cluster
conda install -c pyg pyg
pip install networkx
```

### Running the code

The file structure is quite simple and straightfoward. ```main.py``` is the file used to run the experiments, you can find the various arguments controlling the runs inside. ```data.py``` generates a given number of random d-regular graphs, ```models.py``` contains the GNN model with various possible architectures, and ```utils.py``` contains the code to compute the Hamiltonians for each problem. As a simple example, you can run the following commands to check that the code runs:

```bash
python main.py --maxcut --epochs 3 # Maximum Cut
python main.py --epochs 3 # Maximum Cut
```