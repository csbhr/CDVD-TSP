# CDVD-TSP

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/csbhr/CDVD-TSP/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-0.4.1-%237732a8)](https://pytorch.org/)

#### [Paper](https://arxiv.org/abs/2004.02501) | [Project Page](https://csbhr.github.io/projects/cdvd-tsp/index.html) | [Discussion](https://github.com/csbhr/CDVD-TSP/issues)
### Cascaded Deep Video Deblurring Using Temporal Sharpness Prior[[1](#user-content-citation-1)]
By [Jinshan Pan](https://jspan.github.io/), [Haoran Bai](https://csbhr.github.io/about), and Jinhui Tang

## Updates
[2021-05-23] Many improvements to code.
- Several warnings and errors were addressed
  - Issues Fixed [#12](https://github.com/csbhr/CDVD-TSP/issues/12), [#22](https://github.com/csbhr/CDVD-TSP/issues/22), [#24](https://github.com/csbhr/CDVD-TSP/issues/24), [#26](https://github.com/csbhr/CDVD-TSP/issues/26)
  - PyTorch 1.8.1, Python 3.8, & Cuda 10.2 has been tested
  - PyTorch 0.4.1, Python 3.7, & Cuda 9.2 should still work
    - However, this hasn't been verified for due to test computer unable to run Cuda 9.2
    - It appears that the graphics card is too new for Cuda 9.2
  - Inference no longer requires gt images [#12](https://github.com/csbhr/CDVD-TSP/issues/12), [#22](https://github.com/csbhr/CDVD-TSP/issues/22), [#26](https://github.com/csbhr/CDVD-TSP/issues/26)
  - Added `if __name__ == '__main__':` to `main.py` to resolve [#24](https://github.com/csbhr/CDVD-TSP/issues/24)
  - `optimizer.step()` was added prior to `lr_scheduler.step()`
    - Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details [Here](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
  - Added align_corners=True parameter to `nn.functional.grid_sample` in `flow_pwc.py` since that was the behavior for PyTorch 0.4.1.
    - This behavior was changed in PyTorch 1.3.0 from `True` to `False`. See more details [Here](https://github.com/pytorch/pytorch/releases/tag/v1.3.0)
  - Windows cannot have a colon in a filename.
    - Changed filenames with date time from `YYYY-MM-DD hh:mm:ss` to `YYYY-MM-DDThhmmss` per [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601)
- Changed `log.txt` and `config.txt` files to have the time added to them so that each new run will have a new file. Especially handy for a `--resume`
- `n_frames_per_video` increased from 100 to 200 in order to take full advantage of all of the frames in the training datasets.
- Images are no longer saved by default during test phase of epoch.
  - There was no way to disable with a switch prior to this change
  - Can be enabled with `--save_images` option
- Now gives an estimate of completion time for the current epoch and all epochs
  - 1st training session estimate does not include the 1st test session duration for the 1st epoch.
  - When using `--resume --load`, start time will be recalculated based on the prior elapsed times
- PDF plots no longer created during 1st epoch due to lack of data
  - L1, HEM, & Total Loss plots are now combined in one plot instead of 3
  - PSNR plot no longer has a legend since it was blank
- Inference will handle border situations like this
  - For a video with 5 frames [1, 2, 3, 4, 5] it will use a list of frames [3, 2, 1, 2, 3, 4, 5, 4, 3]
  - Previous handling of the same clip would produce a singe frame [3]
  - The new result will be frames [1, 2, 3, 4, 5] with all 5 frames being deblurred

[2020-10-22] Inference results on DVD and GOPRO are available [[Here]](https://drive.google.com/drive/folders/1lMpj-fkT89JfMOvTnqXMzfa57zCT8-2s?usp=sharing)!  
[2020-10-10] Metrics(PSNR/SSIM) calculating codes are available [[Here]](https://github.com/csbhr/OpenUtility#chapter-calculating-metrics)!  
[2020-08-04] Inference logs are available [[Here]](https://drive.google.com/drive/folders/1lMpj-fkT89JfMOvTnqXMzfa57zCT8-2s?usp=sharing)!  
[2020-03-07] Paper is available!  
[2020-03-31] We further train the model to convergence, and the pretrained model is available!  
[2020-03-07] Add training code!  
[2020-03-04] Testing code is available!

## Experimental Results
Our algorithm is motivated by the success of variational model-based methods. It explores sharpness pixels from adjacent frames by a temporal sharpness prior (see (f)) and restores sharp videos by a cascaded inference process. As our analysis shows, enforcing the temporal sharpness prior in a deep convolutional neural network (CNN) and learning the deep CNN by a cascaded inference manner can make the deep CNN more compact and thus generate better-deblurred results than both the CNN-based methods [27, 32] and variational model-based method [12].  
![top-result](https://s1.ax1x.com/2020/03/31/GQnfpt.png)  

We further train the proposed method to convergence, and get higher PSNR/SSIM than the result reported in the paper.  

Quantitative results on the benchmark dataset by Su et al. [24]. All the restored frames instead of randomly selected 30 frames from each test set [24] are used for evaluations. *Note that: Ours * is the result that we further trained to convergence, and Ours is the result reported in the paper.*  
![table-1](https://s1.ax1x.com/2020/03/31/GQOAv6.png)  

Quantitative results on the GOPRO dataset by Nah et al.[20].  
![table-2](https://s1.ax1x.com/2020/03/31/GQYZi8.png)  

More detailed analysis and experimental results are included in [[Project Page]](https://csbhr.github.io/projects/cdvd-tsp/index.html).

## Dependencies

- We use a modified version of the implementation of PWC-Net[[3](#user-content-citation-3)] by [[sniklaus/pytorch-pwc]](https://github.com/sniklaus/pytorch-pwc)
- Linux (Tested on Ubuntu 18.04)
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch 0.4.1](https://pytorch.org/): `conda install pytorch=0.4.1 torchvision cudatoolkit=9.2 -c pytorch`
- numpy: `conda install numpy`
- matplotlib: `conda install matplotlib`
- opencv: `conda install opencv`
- imageio: `conda install imageio`
- skimage: `conda install scikit-image`
- tqdm: `conda install tqdm`
- cupy: `conda install -c anaconda cupy`
### Example for Python 3.7
```dos
conda install -y pytorch=0.4.1 torchvision cudatoolkit=9.2 -c pytorch
conda install -y matplotlib
conda install -y opencv
conda install -y imageio
conda install -y scikit-image
conda install -y tqdm
pip install cupy-cuda92
```
### Example for Python 3.8
```dos
conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -y matplotlib
conda install -y opencv
conda install -y imageio
conda install -y scikit-image
conda install -y tqdm
pip install cupy-cuda102
```


## Get Started

### Download
- Pretrained models and Datasets can be downloaded [[Here]](https://drive.google.com/drive/folders/1lw_1jITafEQ9DvMys_S6aYwtNApYKWsz?usp=sharing).
	- If you have downloaded the pretrained models，please put them to './pretrain_models'.
      - CDVD_TSP_DVD_Paper.pt pretrained model that was done for the paper using [DeepVideoDeblurring dataset](https://github.com/shuochsu/DeepVideoDeblurring)
      - CDVD_TSP_GOPRO.pt pretrained model using [GOPRO_Large dataset](https://github.com/SeungjunNah/DeepDeblur_release)
      - CDVD_TSP_DVD_Convergent.pt pretrained model that continued until convergence
      - network-default.pytorch pretrained model for PWC-Net[[3](#user-content-citation-3)]
        - Available from the PyTorch-PWC author and can be downloaded [[Here]](http://content.sniklaus.com/github/pytorch-pwc/network-default.pytorch).
	- If you have downloaded the datasets，please put them to './dataset'.

### Dataset Organization Form
If you prepare your own dataset, please follow the following form:
```
|--dataset  
    |--blur  
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
            :
        |--video n
    |--gt
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
        	:
        |--video n
```

### Training
- Download the PWC-Net[[3](#user-content-citation-3)] pretrained model.
- Download training dataset, or prepare your own dataset like above form.
- Each epoch has a training phase and a testing phase to determine the PSNR and evaluate the results.
- Run the following commands:
```
cd ./code
python main.py --save Experiment_Name --dir_data path/to/train/dataset --dir_data_test path/to/val/dataset --epochs 500 --batch_size 8
  # --save:          Experiment name to save
                     The experiment result will be in './experiment/'.
  # --dir_data:      The path of the training dataset.
                     Used during the training phase of the epoch.
  # --dir_data_test: The path of the evaluating dataset during training process.
                     Used during the testing phase of the epoch.
  # --epochs:        The number of training epochs.
  # --batch_size:    The mini batch size.
  # --print_every:   How many batches to wait before logging training status
                     during training phase of epoch.
  # --save_images:   Save images during test phase of epoch.
  # --load:          Experiment name to load
                     The experiment result must be in './experiment/'.
                     Use --resume to continue at the last epoch completed
  # --resume:        Resume from the latest complete epoch must use --load instead of --save.
```
#### Examples of Training
`python main.py --save CDVD_TSP_DVD_Convergent --dir_data F:\workspaces\CDVD-TSP\dataset\DVD\train --dir_data_test F:\workspaces\CDVD-TSP\dataset\DVD\test --epochs 1000 --print_every 500 --batch_size 2`

`python main.py --resume --load CDVD_TSP_DVD_Convergent --dir_data F:\workspaces\CDVD-TSP\dataset\DVD\train --dir_data_test F:\workspaces\CDVD-TSP\dataset\DVD\test --epochs 1000 --print_every 500 --batch_size 2`


### Testing

#### Quick Test
- Download the pretrained models.
- Download the testing dataset.
- Run the following commands:
```
cd ./code
python inference.py --default_data DVD
	# --default_data: the dataset you want to test, optional: DVD, GOPRO
```
- The deblured result will be in './infer_results'.

#### Test Your Own Dataset
- Download the pretrained models.
- Organize your dataset like the above form.
- Run the following commands:
```
cd ./code
python inference.py --data_path path/to/data --model_path path/to/pretrained/model
	# --data_path: the path of your dataset.
	# --model_path: the path of the downloaded pretrained model.
```
- The deblured result will be in './infer_results'.

## Citations
### Citation 1
```
@InProceedings{Pan_2020_CVPR,
    author = {Pan, Jinshan and Bai, Haoran and Tang, Jinhui},
    title = {Cascaded Deep Video Deblurring Using Temporal Sharpness Prior},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```
### Citation 2
```
@misc{CDVD-TSP,
    author = {Haoran Bai},
    title = {Cascaded Deep Video Deblurring Using Temporal Sharpness Prior},
    year = {2020},
    howpublished = {\url{https://github.com/csbhr/CDVD-TSP}}
}
```
### Citation 2
```
@misc{pytorch-pwc,
    author = {Simon Niklaus},
    title = {A Reimplementation of {PWC-Net} Using {PyTorch}},
    year = {2018},
    howpublished = {\url{https://github.com/sniklaus/pytorch-pwc}}
}
```
