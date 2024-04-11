# behavioral_prediction_588
scripts for behavioral prediction pipeline

# Setting up the environment for AgentFormer
- Run `git clone https://github.com/Khrylx/AgentFormer.git`. This should create a folder called AgentFormer.
- Create the following `environment.yml` file inside the AgentFormer folder:
```
name: AgentFormer
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pytorch=1.8.0
  - torchvision=0.9.0
  - torchaudio=0.8.0
  - cudatoolkit=11.1
  - pip
  - pip:
    - -r requirements.txt
```
- Create the environment from the file with the command: `conda env create -f environment.yml`
- Activate the environment with the command: `conda activate AgentFormer`
- Note that in `/data/preprocessor.py` on line 37, you need to change `np.int` to `np.int64` since the former is a depricated form of representing an integer in numpy.
