# PCR: A Parallel Convolution Residual Network For Traffic Flow Prediction

This is a pytorch implementation of 《PCR: A Parallel Convolution Residual Network For Traffic Flow Prediction》(IEEE TETCI 2024, under review). 

Detail information will be released after publication.


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
    <td></td>
    <td></td>
    <td>RMSE</td>
    <td>MAE</td>
    <td>RMSE</td>
    <td>MAE</td>
    <td>RMSE</td>
    <td>MAE</td>
  </tr>
  <tr>
    <td>HA</td>
    <td></td>
    <td>41.81</td>
    <td>22.85</td>
    <td>10.12</td>
    <td>5.02</td>
    <td>57.42</td>
    <td>25.42</td>
  </tr>
</table>

\begin{table}[h]
\begin{center}
\caption{COMPARISON WITH BASELINE METHODS ON THREE DATASETS}
\resizebox{\linewidth}{!}{\begin{tabular}{cccccccc}
\hline
\multirow{2}{*}{Type}  & \multirow{2}{*}{Model} & \multicolumn{2}{c}{TaxiBJ}                       & \multicolumn{2}{c}{BikeNYC}                     & \multicolumn{2}{c}{TaxiCQ}                     \\ \cline{3-8} 
                       &                        & RMSE                    & MAE                    & RMSE                   & MAE                    & RMSE                 & MAE                     \\ \hline
\multirow{2}{*}{Basic} & HA                     &                    &                   &                   &                   &                &                    \\
                       & GRU                    & 17.76                   & 10.01                  & 8.48                   & 4.06                   & 33.45                & 19.16                   \\ \hline
GCN                    & STSSL                  & 18.62                   & 11.42                  & 7.61                   & 5.09                   & 49.17                & 27.9                    \\ \hline
\multirow{6}{*}{Attention}              & MSP-STTN               & 16.44                   & 9.28                   & 5.92                   & 2.98                   & 44.57                & 21.04                   \\
                       & ASTCN                  & 16.14                   & 9.38                   & \textbf{4.65} & 2.44                   & 21.97                & 13.46                   \\
                       & ACFM                   & 16.07                   & 9.34                   & 5.73                   & 2.86                   & 27.54                & 13.79                   \\
\multicolumn{1}{l}{}   & MCSTL                  & 16.03                   & 9.28                   & 5.17                   & 2.52                   & 23.83                & 16.47                   \\
                       & ST-GSP                 & 16.03                   & 9.26                   & 5.81                   & 2.92                   & 31.11                & 15.4                    \\
                       & RATFM                  & 16.01                   & 9.24                   & 5.48                   & 2.75                   & 24.1                 & 17.62                   \\ \hline
\multirow{4}{*}{CNN}   & MHSRN                  & 17.54                   & 10.45                  & 4.79                   & \textbf{2.23} & 23.33                & 13.91                   \\
                       & STResNet               & 16.69                   & 9.46                   & 6.33                   & 2.94                   & 26.16                & 14.97                   \\
                       & LMST3D-ResNet          & 16.37                   & 9.36                   & 5.13                   & 2.48                   & 34.96                & 16.53                   \\
                       & PCR(Ours)                    & \textbf{15.94} & \textbf{9.23} & 4.99                   & 2.43                   & \textbf{20} & \textbf{12.21} \\ \hline
\end{tabular}
}
\label{table3}
\end{center}
\end{table}

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

- train_model.py: The training code of our model
- utils.py: Dataset code
- Model.py: main file for Model
- requirements.txt: List the pip dependencies

## Dependency

After installing the dependency:

    pip install -r requirements.txt

## Train

- Download datasets from [here](https://github.com/CQRhinoZ/TaxiCQ).
- Run `train_model.py`.


## Citation

```
<TBD>
```

Feel free to contact us:

Xu ZHANG, Ph.D, Professor

Chongqing University of Posts and Telecommunications

Email: zhangx@cqupt.edu.cn

Website: https://faculty.cqupt.edu.cn/zhangx/zh_CN/index.htm
