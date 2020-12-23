# GradSLAM-RGB-D-Completion
 Leveraging GradSLAM Multi-view gradients to optimize RGB-D Images: Experiments and Insights
 
## Experiments

### Constant Value RGB-D Image

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/constant.png" width="40" height="40" />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/s_constant.png" width="40" height="40" />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/gifs/cv_d.gif" width="40" height="40" />
<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/gifs/cv_r.gif" width="40" height="40" />

### Uniform Noise Image

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/uniform.png" width="40" height="40" />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/s_uniform.png" width="40" height="40" />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/gifs/un_d.gif" width="40" height="40" />
<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/gifs/un_r.gif" width="40" height="40" />

### Semantic Adversarial attack on Gt RGB-D Image

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/semantic.png" width="40" height="40" />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/s_semantic.png" width="40" height="40" />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/gifs/sem_d.gif" width="40" height="40" />
<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/gifs/sem_r.gif" width="40" height="40" />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/sem.png" width="40" height="40" />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/recon_sem.png" width="40" height="40" />

### Slight Noise addition to Gt RGB-D Image

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/slight.png"  width="40" height="40" />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/s_slight.png" width="40" height="40" />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/gifs/sn_d.gif" width="40" height="40" />
<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/gifs/sn_r.gif" width="40" height="40" />

### Salt & Pepper Noise RGB Image with Gt Depth

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/s&p.png" width="40" height="40" />


## Insights & Discussion

## Running Experiments on Colab

``` 
!pip install -q 'git+https://github.com/gradslam/gradslam.git' 
!git clone https://github.com/NikV-JS/GradSLAM-RGB-D-Completion.git
%cd /content/GradSLAM-RGB-D-Completion
!python main.py --save_dir='/content/' --experiment
```

## Data & Code Archive

## Conclusion & Further Study

## Acknowledgements
