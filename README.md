# Repeatable and Reliable Detector and Descriptor with different losses #

This repository is made as part of the assignment for Seminar Computer Vision by Deep Learning (CS4245) at TUDelft. A [blog](https://hackmd.io/mKx4o2N2QKCs6qXL6aQxlw) is made in parallel.

**Author:Ranbao Deng, (5460069), R.Deng@student.tudelft.nl**
**Author:Danning Zhao, (5459583), D.Zhao-3@student.tudelft.nl**

This repository is based on [R2D2](https://github.com/naver/r2d2) and [D2-Net](https://github.com/mihaidusmanu/d2-net). We focused on the trainer, nets, losses and visualizations. To see our results, please follow the Getting stated and Evaluation on HPatches part.

This repository contains the implementation of the following [paper](https://europe.naverlabs.com/research/publications/r2d2-reliable-and-repeatable-detectors-and-descriptors-for-joint-sparse-local-keypoint-detection-and-feature-extraction/):

```text
@inproceedings{r2d2,
  author    = {Jerome Revaud and Philippe Weinzaepfel and C{\'{e}}sar Roberto de Souza and
               Martin Humenberger},
  title     = {{R2D2:} Repeatable and Reliable Detector and Descriptor},
  booktitle = {NeurIPS},
  year      = {2019},
}
```


Getting started
---------------
You just need Python 3.6+ equipped with standard scientific packages and PyTorch1.1+.
Typically, conda is one of the easiest way to get started:
```bash
conda install python tqdm pillow numpy matplotlib scipy
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```


Models
-----------------
For your convenience, we provide five pre-trained models in the `models/` folder:
 - `r2d2_WAF_N16.pt`: this is the model used in most experiments of the paper (on HPatches `MMA@3=0.686`). It was trained with Web images (`W`), Aachen day-time images (`A`) and Aachen optical flow pairs (`F`)
 - `r2d2_WASF_N16.pt`: this is the model used in the visual localization experiments (on HPatches `MMA@3=0.721`). It was trained with Web images (`W`), Aachen day-time images (`A`), Aachen day-night synthetic pairs (`S`), and Aachen optical flow pairs (`F`).
 - `r2d2_WASF_N8_big.pt`: Same than previous model, but trained with `N=8` instead of `N=16` in the repeatability loss. In other words, it outputs a higher density of keypoints. This can be interesting for certain applications like visual localization, but it implies a drop in MMA since keypoints gets slighlty less reliable.
 - `faster2d2_WASF_N16.pt`: The Fast-R2D2 equivalent of r2d2_WASF_N16.pt
 - `faster2d2_WASF_N8_big.pt`: The Fast-R2D2 equivalent of r2d2_WASF_N8.pt
  
For more details about the training data, see the dedicated section below.
Here is a table that summarizes the performance of each model:

|    model name    | model size<br>(#weights)| number of<br>keypoints |MMA@3 on<br>HPatches|
|------------------|:-----------------------:|:----------------------:|:------------------:|
|`r2d2_WAF_N16.pt`    | 0.5M                    | 5K                     | 0.686              |
|`r2d2_WASF_N16.pt`   | 0.5M                    | 5K                     | 0.721              |
|`r2d2_WASF_N8_big.pt`| 1.0M                    | 10K                    | 0.692              |
|`faster2d2_WASF_N8_big.pt`| 1.0M                    | 5K                    | 0.650              |
<!--|`r2d2_WASF_N8_big.pt`| 1.0M                    | 5K                     | 0.704              |-->

Our models are in the `models/new models` directory. They have the same hyper-parameters as the `faster2d2_WASF_N16.pt`.

Feature extraction
------------------
To extract keypoints for a given image, simply execute:
```bash
python extract.py --model models/r2d2_WASF_N16.pt --images imgs/brooklyn.png --top-k 5000
```
This also works for multiple images (separated by spaces) or a `.txt` image list. 
For each image, this will save the `top-k` keypoints in a file with the same path as the image and a `.r2d2` extension. 
For example, they will be saved in `imgs/brooklyn.png.r2d2` for the sample command above.

The keypoint file is in the `npz` numpy format and contains 3 fields: 
 - `keypoints` (`N x 3`): keypoint position (x, y and scale). Scale denotes here the patch diameters in pixels.
 - `descriptors` (`N x 128`): l2-normalized descriptors.
 - `scores` (`N`): keypoint scores (the higher the better). 
 
*Note*: You can modify the extraction parameters (scale factor, scale range...). Run `python extract.py --help` for more information. 
By default, they corespond to what is used in the paper, i.e., a scale factor equal to `2^0.25` (`--scale-f 1.189207`) and image size in the range `[256, 1024]` (`--min-size 256 --max-size 1024`). 

*Note2*: You can significantly improve the `MMA@3` score (by ~4 pts) if you can afford more computations. To do so, you just need to increase the upper-limit on the scale range by replacing `--min-size 256 --max-size 1024` with `--min-size 0 --max-size 9999 --min-scale 0.3 --max-scale 1.0`.

Evaluation on HPatches
----------------------
The evaluation is based on the [code](https://github.com/mihaidusmanu/d2-net) from [D2-Net](https://dsmn.ml/publications/d2-net.html).
```bash
git clone https://github.com/mihaidusmanu/d2-net.git
cd d2-net/hpatches_sequences/
bash download.sh
bash download_cache.sh
cd ../..
ln -s d2-net/hpatches_sequences # create a soft-link
ln -s d2-net/qualitative # create a soft-link
```

Once this is done, extract all the features:
```bash
python extract.py --model models/r2d2_WAF_N16.pt --images d2-net/image_list_hpatches_sequences.txt
```

Finally, evaluate using the iPython notebook `d2-net/hpatches_sequences/HPatches-Sequences-Matching-Benchmark.ipynb`.
Qualitative evaluations could be done using the iPython notebook `d2-net/qualitative/Qualitative-Matches.ipynb`.

In order to see our result, please replace the above two files with `visualization/HPatches-Sequences-Matching-Benchmark.ipynb` and `visualization/Qualitative-Matches.ipynb`.

The figures could be seen in `figs/`.

Here is a summary of the results:

| Variants                 | MMA@1 | MMA@3 | MMA@6 | MMA@10 |
|--------------------------|-------|-------|-------|--------|
| Original (faster R2D2)   | 0.278 | 0.700 | 0.837 | 0.865  |
| Original f0              | 0.282 | 0.703 | 0.837 | 0.865  |
| Hardnet reliability      | 0.281 | 0.704 | 0.838 | 0.865  |
| Hardnet reliability f0   | 0.283 | 0.704 | 0.836 | 0.864  |
| Hardnet with dice        | 0.279 | 0.705 | 0.840 | 0.868  |
| Czekanowski similarity   | 0.285 | 0.709 | 0.840 | 0.866  |
| Dice similarity          | 0.289 | 0.715 | 0.846 | 0.872  |
| Jaccard similarity       | 0.281 | 0.706 | 0.841 | 0.868  |
| Squared-chord similarity | 0.269 | 0.664 | 0.787 | 0.812  |

Training the model
------------------
We provide all the code and data to retrain the model as described in the paper.

### Downloading training data ###
The first step is to download the training data. 
First, create a folder that will host all data in a place where you have sufficient disk space (15 GB required).
```bash
DATA_ROOT=/path/to/data
mkdir -p $DATA_ROOT
ln -fs $DATA_ROOT data 
mkdir $DATA_ROOT/aachen
```
Then, manually download the [Aachen dataset here](https://drive.google.com/drive/folders/1fvb5gwqHCV4cr4QPVIEMTWkIhCpwei7n) and save it as `$DATA_ROOT/aachen/database_and_query_images.zip`.
Finally, execute the download script to complete the installation. It will download the remaining training data and will extract all files properly.
```bash 
./download_training_data.sh
```
The following datasets are now installed:

| full name                       |tag|Disk |# imgs|# pairs| python instance                |
|---------------------------------|---|-----|------|-------|--------------------------------|
| Random Web images               | W |2.7GB| 3125 |  3125 | `auto_pairs(web_images)`       |
| Aachen DB images                | A |2.5GB| 4479 |  4479 | `auto_pairs(aachen_db_images)` |
| Aachen style transfer pairs     | S |0.3GB| 8115 |  3636 | `aachen_style_transfer_pairs`  |
| Aachen optical flow pairs       | F |2.9GB| 4479 |  4770 | `aachen_flow_pairs`            |

Note that you can visualize the content of each dataset using the following command:
```bash
python -m tools.dataloader "PairLoader(aachen_flow_pairs)"
```
![image](https://user-images.githubusercontent.com/56719813/68311498-eafecd00-00b1-11ea-8d37-6693f3f90c9f.png)


### Training details ###
To train the model, simply run this command:
```bash
python train.py --save-path /path/to/model.pt 
```
On a GTX1070m, it takes 80 min per epoch, so ~7h for 5 epochs.

Note that you can fully configure the training (i.e. select the data sources, change the batch size, learning rate, number of epochs etc.). One easy way to improve the model is to train for more epochs, e.g. `--epochs 50`. For more details about all parameters, run `python train.py --help`.
