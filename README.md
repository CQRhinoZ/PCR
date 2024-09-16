# PCR: A Parallel Convolution Residual Network For Traffic Flow Prediction

This is a pytorch implementation of 《PCR: A Parallel Convolution Residual Network For Traffic Flow Prediction》(IEEE TETCI 2024, under review). 


## Abstract

Traffic flow prediction is crucial in smart cities and traffic management, yet it presents challenges due to intricate spatial-temporal dependencies and external factors. Most existing research relied on a traditional data selection approach to represent temporal dependence. However, only considering spatial dependence in adjacent or distant regions limits the performance. In this paper, we propose an end-to-end Parallel Convolution Residual network (PCR) for grid-based traffic flow prediction. Firstly, we introduce a novel data selection strategy to capture more temporal dependence, and then we implement an early fusion strategy without any additional operations to obtain a lighter model. Secondly, we propose to extract external features with feature embedding matrix operations, which can represent the interrelationships between different kinds of external data. Finally, we build a parallel residual network with concatenated features as input, which is composed of a standard residual net (SRN) to extract short spatial dependence and a dilated residual net (DRN) to extract long spatial dependence. Experiments on three traffic flow datasets TaxiBJ, BikeNYC, and TaxiCQ exhibit that the proposed method outperforms the state-of-the-art models with the most minor parameters.

## Performance
TABLE II: PERFORMANCE COMPARISON IN ACC@K ON THREE DATASETS
<table style="width:100%;">
  <tr>
    <th rowspan="2">Type</th>
    <th rowspan="2">Model</th>
    <th colspan="2">TaxiBJ</th>
    <th colspan="2">BikeNYC</th>
    <th colspan="2">TaxiCQ</th>
  </tr>
  <tr>
    <td>RMSE</td>
    <td>MAE</td>
    <td>RMSE</td>
    <td>MAE</td>
    <td>RMSE</td>
    <td>MAE</td>
  </tr>
  <tr>
    <th rowspan="2">Basic</td>
    <td>HA</td>
    <td>41.81</td>
    <td>22.85</td>
    <td>10.12</td>
    <td>5.02</td>
    <td>57.42</td>
    <td>25.42</td>
  </tr>
  <tr>
    <td>GRU</td>
    <td>17.76</td>
    <td>10.01</td>
    <td>8.48</td>
    <td>4.06 </td>
    <td>33.45</td>
    <td>19.16</td>
  </tr>
  <tr>
    <td>GCN</td>
    <td>STSSL</td>
    <td>18.62</td>
    <td>11.42</td>
    <td>7.61</td>
    <td>5.09</td>
    <td>49.17</td>
    <td>27.9</td>
  </tr>
  <tr>
    <th rowspan="6">Attention</td>
    <td>MSP-STTN</td>
    <td>16.44</td>
    <td>9.28</td>
    <td>5.92</td>
    <td>2.98</td>
    <td>44.57</td>
    <td>21.04</td>
  </tr>
  <tr>
    <td>ASTCN</td>
    <td>16.14</td>
    <td>9.38</td>
    <td>4.65</td>
    <td>2.44</td>
    <td>21.97</td>
    <td>13.46</td>
  </tr>
  <tr>
    <td>ACFM</td>
    <td>16.07</td>
    <td>9.34</td>
    <td>5.73</td>
    <td>2.86</td>
    <td>27.54</td>
    <td>13.79</td>
  </tr>
  <tr>
    <td>MCSTL</td>
    <td>16.03</td>
    <td>9.28</td>
    <td>5.17</td>
    <td>2.52</td>
    <td>23.83</td>
    <td>16.47</td>
  </tr>
  <tr>
    <td>ST-GSP</td>
    <td>16.03</td>
    <td>9.26</td>
    <td>5.81</td>
    <td>2.92</td>
    <td>31.11</td>
    <td>15.4</td>
  </tr>
  <tr>
    <td>RATFM</td>
    <td>16.01</td>
    <td>9.24</td>
    <td>5.48</td>
    <td>2.75</td>
    <td>24.1</td>
    <td>17.62</td>
  </tr>
  <tr>
    <th  rowspan="4">CNN</td>
    <td>MHSRN</td>
    <td>17.54</td>
    <td>10.45</td>
    <td>4.79</td>
    <td>2.23</td>
    <td>23.33</td>
    <td>13.91</td>
  </tr>
  <tr>
    <td>STResNet</td>
    <td>16.69</td>
    <td>9.46</td>
    <td>6.33</td>
    <td>2.94</td>
    <td>26.16</td>
    <td>14.97</td>
  </tr>
  <tr>
    <td>LMST3D-ResNet</td>
    <td>16.37</td>
    <td>9.36</td>
    <td>5.13</td>
    <td>2.48</td>
    <td>34.96</td>
    <td>16.53</td>
  </tr>
  <tr>
    <td>PCR(Ours)</td>
    <td>15.94</td>
    <td>9.23</td>
    <td>4.99</td>
    <td>2.43</td>
    <td>20</td>
    <td>12.21</td>
  </tr>
</table>


## Architecture

```
<TBD>
```

## Installation

- Install Pytorch 1.8.1 (Note that the results reported in the paper are obtained by running the code on this Pytorch version. As raised by the issue, using higher version of Pytorch may seem to have a performance decrease on optic cup segmentation.)
- Clone this repo

```
git clone https://github.com/CQRhinoZ/PCR
```

## Project Structure

- Train_TaxiBJ.py: The training code of our model
- load_test.py: The testing code for TaxiBJ
- model.py: main file for Model
- requirements.txt: List the pip dependencies

## Dependency

After installing the dependency:

    pip install -r requirements.txt

## Train

- Download datasets from [here](https://drive.google.com/drive/folders/1sbVO27r6zWedQ_UyruiXOYDjbdKWsBSj?usp=drive_link).
- Run `Train_TaxiBJ.py`.
  

## Citation

```
<TBD>
```

Feel free to contact us:

Xu ZHANG, Ph.D, Professor

Chongqing University of Posts and Telecommunications

Email: zhangx@cqupt.edu.cn

Website: https://faculty.cqupt.edu.cn/zhangx/zh_CN/index.htm
