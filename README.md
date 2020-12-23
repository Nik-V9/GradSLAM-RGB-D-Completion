# GradSLAM-RGB-D-Completion
 Leveraging GradSLAM Multi-view gradients to optimize RGB-D Images: Experiments and Insights
 
## Experiments

### Constant Value RGB-D Image

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/constant.png" />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/s_constant.png" />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/gifs/cv_d.gif" />
<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/gifs/cv_r.gif"  />

### Uniform Noise Image

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/uniform.png" />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/s_uniform.png" />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/gifs/un_d.gif"  />
<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/gifs/un_r.gif" />

### Semantic Adversarial attack on Gt RGB-D Image

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/semantic.png" />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/s_semantic.png"  />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/gifs/sem_d.gif"  />
<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/gifs/sem_r.gif"  />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/sem.png" />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/recon_sem.png"  />

### Slight Noise addition to Gt RGB-D Image

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/slight.png"   />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/s_slight.png"  />

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/gifs/sn_d.gif"  />
<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/gifs/sn_r.gif"  />

### Salt & Pepper Noise RGB Image with Gt Depth

<img src="https://github.com/NikV-JS/GradSLAM-RGB-D-Completion/blob/main/images/s&p.png"  />


## Insights & Discussion

## Conclusion & Further Study

## Running Experiments on Colab

``` 
!pip install -q 'git+https://github.com/gradslam/gradslam.git' 
!git clone https://github.com/NikV-JS/GradSLAM-RGB-D-Completion.git
%cd /content/GradSLAM-RGB-D-Completion
!python main.py --save_dir='/content/' --experiment
```

## Data & Code Archive

## Acknowledgements
