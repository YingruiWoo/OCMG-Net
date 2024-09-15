# OCMG-Net: Neural Oriented Normal Refinement for Unstructured Point Clouds

### *[ArXiv](https://arxiv.org/abs/2409.01100) 

We present a robust refinement method for estimating oriented normals from unstructured point clouds. In contrast to previous approaches that either suffer from high computational complexity or fail to achieve desirable accuracy, our novel framework incorporates sign orientation and data augmentation in the feature space to refine the initial oriented normals, striking a balance between efficiency and accuracy. To address the issue of noise-caused direction inconsistency existing in previous approaches, we introduce a new metric called the Chamfer Normal Distance, which faithfully minimizes the estimation error by correcting the annotated normal with the closest point found on the potentially clean point cloud. This metric not only tackles the challenge but also aids in network training and significantly enhances network robustness against noise. Moreover, we propose an innovative dual-parallel architecture that integrates Multi-scale Local Feature Aggregation and Hierarchical Geometric Information Fusion, which enables the network to capture intricate geometric details more effectively and notably reduces ambiguity in scale selection. Extensive experiments demonstrate the superiority and versatility of our method in both unoriented and oriented normal estimation tasks across synthetic and real-world datasets among indoor and outdoor scenarios. This project is the implementation of OCMG-Net by Pytorch.

## Requirements
The code is implemented in the following environment settings:
- Ubuntu 20.04
- CUDA 11.3
- Python 3.8
- Pytorch 1.12
- Numpy 1.24
- Scipy 1.10
- Open3D 0.18

## Dataset
We train our network model on the [PCPNet](http://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/pclouds.zip) dataset.
Download the dataset to the folder `***/dataset/` and copy the list into the fold `***/dataset/PCPNet/list`. 
More test datasets with complex geometries can be downloaded from [here](https://drive.google.com/drive/folders/1eNpDh5ivE7Ap1HkqCMbRZpVKMQB1TQ6H?usp=share_link).
The dataset is organized as follows:
```
│dataset/
├──PCPNet/
│  ├── list
│      ├── ***.txt
│  ├── pre_oriented
│      ├── ***_oriented.npy
│  ├── ***.xyz
│  ├── ***.normals
│  ├── ***.pidx
├──FamousShape/
│  ├── list
│      ├── ***.txt
│  ├── pre_oriented
│      ├── ***_oriented.npy
│  ├── ***.xyz
│  ├── ***.normals
│  ├── ***.pidx
```

## Train
Our trained model is provided in `./log/001/ckpts/ckpt_900.pt`.
To train a new model on the PCPNet dataset, simply run:
```
python train.py
```
Your trained model will be save in `./log/***/ckpts/`.

## Test
Our pre-processed data can be downloaded from [here](https://drive.google.com/drive/folders/1ZqyaSq1rUznPfjGiN0hpTWqvt5TCRRRg?usp=sharing).
You can use the provided model for testing:
- PCPNet dataset
```
python test.py --data_set='PCPNet'  --testset_list='testset_all.txt', --eval_list=['testset_no_noise.txt', 'testset_low_noise.txt', 'testset_med_noise.txt', 'testset_high_noise.txt', 'testset_vardensity_striped.txt', 'testset_vardensity_gradient.txt']
```
The evaluation results will be saved in `./log/001/results_PCPNet/ckpt_900/`.
- FamousShape dataset
```
python test.py --data_set='FamousShape'  --testset_list='testset_FamousShape', --eval_list=['testset_noise_clean.txt', 'testset_noise_low.txt', 'testset_noise_med.txt', 'testset_noise_high.txt', 'testset_density_stripe.txt', 'testset_density_gradient.txt']
```
The evaluation results will be saved in `./log/001/results_FamousShape/ckpt_900/`.

To test with your trained model, you need to change the variables in `test.py`:
```
ckpt_dirs       
ckpt_iter
```
To save the normals of the input point cloud, you need to change the variables in `test.py`:
```
save_pn = True        # to save the point normals as '.normals' file
sparse_patches = False  # to output sparse point normals or not
```

## Acknowledgement
The code is heavily based on [SHS-Net](https://github.com/LeoQLi/SHS-Net).
If you find our work useful in your research, please cite the following papers:

```
@article{wu2024ocmg,
  title={OCMG-Net: Neural Oriented Normal Refinement for Unstructured Point Clouds},
  author={Wu, Yingrui and Zhao, Mingyang and Quan, Weize and Shi, Jian and Jia, Xiaohong and Yan, Dong-Ming},
  journal={arXiv preprint arXiv:2409.01100},
  year={2024}
}

@inproceedings{wu2024cmg,
  title={CMG-Net: Robust Normal Estimation for Point Clouds via Chamfer Normal Distance and Multi-Scale Geometry},
  author={Wu, Yingrui and Zhao, Mingyang and Li, Keqiang and Quan, Weize and Yu, Tianqi and Yang, Jianfeng and Jia, Xiaohong and Yan, Dong-Ming},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={6},
  pages={6171--6179},
  year={2024}
}

@inproceedings{li2023shs,
  title={SHS-Net: Learning signed hyper surfaces for oriented normal estimation of point clouds},
  author={Li, Qing and Feng, Huifang and Shi, Kanle and Gao, Yue and Fang, Yi and Liu, Yu-Shen and Han, Zhizhong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13591--13600},
  year={2023}
}
```

