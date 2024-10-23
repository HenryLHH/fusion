## FUSION: Safety-aware Causal Representation for Trustworthy Offline Reinforcement Learning in Autonomous Driving

#### [[Project Page]](https://sites.google.com/view/safe-fusion/) | [[RAL Paper]](https://ieeexplore.ieee.org/document/10476686) | [[Arxiv Preprint]](https://arxiv.org/pdf/2311.10747) | [[Dataset]](https://drive.google.com/drive/folders/10T-i_SlHRkB5FKLCa1BO4rpZgw-9x3AN) | [[Video]](https://drive.google.com/file/d/1lV1NWw0D1nH1-KjX1RqH4kDcaFSIJM1i/view)

![[Diagram Preview]](fusion/diagram.jpg)

### 🔍 Structure
The core structure of this repo is as follows:
```
├── fusion
│   ├── agent       # the training configs of each algorithm
│   ├── configs     # the evaluation escipts
│   ├── envs        # the training and testing environments
├── utils           # the globally shared utility functions
├── train           # the training scripts
├── tools           # tools for debugging and visualization
├── scripts
```
The implemented offline safe RL and imitation learning algorithms include:

| Algorithm           | Type           | Description           |
|:-------------------:|:-----------------:|:------------------------:|
| BC               | Imitation Learning | [Behavior Cloning](https://arxiv.org/abs/2302.07351) |
| ICIL            | Imitation Learning           | [Invariant Causal Imitation Learning](https://arxiv.org/pdf/1812.02900.pdf)|
| GSA            | Imitation Learning           | [Generalized State Abstraction](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8594201) |
| BNN            | Imitation Learning           | [Ensemble Bayesian Decision Making](https://arxiv.org/pdf/1811.12555) |
| BEAR-Lag            | Offline Safe RL           | [BEARL](https://arxiv.org/abs/1906.00949) with [PID Lagrangian](https://arxiv.org/abs/2007.03964)   |
| BCQ-Lag             | Offline Safe RL          | [BCQ](https://arxiv.org/pdf/1812.02900.pdf) with [PID Lagrangian](https://arxiv.org/abs/2007.03964) |
| CPQ                 | Offline Safe RL           | [Constraints Penalized Q-learning (CPQ)](https://arxiv.org/abs/2107.09003) |


### 📝 Guidelines

#### Installation

```shell
git clone https://github.com/HenryLHH/fusion

# install fusion package in the virtual env
cd fusion
conda create -n fusion
conda activate fusion
pip install -e .[all]
```

#### Training and Evaluation

```shell
# train the FUSION agents
bash scripts/run_fusion.sh
# train the other agents
bash scripts/run_all.sh
# evaluate the trained model
bash scripts/run_eval_task.sh
# visualization
bash scripts/run_vis.sh
```
------------

### 💾 Data Availability

Our dataset to train the offline RL and imitation learning baselines is available on this [Google Drive Link](https://drive.google.com/drive/folders/10T-i_SlHRkB5FKLCa1BO4rpZgw-9x3AN?usp=sharing). 

### ❤️ Acknowledgement 

We acknowledge the following related repositories which contributes to some of our baselines in this offline RL and imitation learning libraries for autonomous driving in metadrive:

- ICIL: https://github.com/ioanabica/Invariant-Causal-Imitation-Learning
- OSRL: https://github.com/liuzuxin/OSRL

### 📚 Reference

For more information about implementation, you are welcome to check our [RAL paper](https://arxiv.org/pdf/2311.10747): 

```bibtex
@article{lin2024safety,
  title={Safety-aware causal representation for trustworthy offline reinforcement learning in autonomous driving},
  author={Lin, Haohong and Ding, Wenhao and Liu, Zuxin and Niu, Yaru and Zhu, Jiacheng and Niu, Yuming and Zhao, Ding},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```
