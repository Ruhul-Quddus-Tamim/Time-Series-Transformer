[![Unit Testing](https://github.com/Ruhul-Quddus-Tamim/Time-Series-Transformer/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/Ruhul-Quddus-Tamim/Time-Series-Transformer/actions/workflows/main.yml)

# Time Series Transformer
Self-attention, the revolutionary concept introduced by the paper [Attention is all you need](https://arxiv.org/abs/1706.03762), is the basis of this foundation model. TimeGPT model is not based on any existing large language model(LLM). Instead, it is independently trained on a vast amount of time series data, and the large transformer model is designed to minimize the forecasting error.

The architecture consists of an encoder-decoder structure with multiple layers, each with residual connections and layer normalization. Finally, a linear layer maps the decoder’s output to the forecasting window dimension. The general intuition is that attention-based mechanisms are able to capture the diversity of past events and correctly extrapolate potential future distributions.

<img width="1319" alt="forecast" src="https://github.com/user-attachments/assets/cffed809-3d13-49e2-a2ed-2451f703201e">

To make prediction, TimeGPT “reads” the input series much like the way humans read a sentence – from left to right. It looks at windows of past data, which we can think of as “tokens”, and predicts what comes next. This prediction is based on patterns the model identifies in past data and extrapolates into the future.

**TimeGPT** is closed source. However, the SDK is open source and available under the Apache 2.0 License.

# 🚀 Building a Time-Series Transformer from Scratch! 🚀
Inspired by the innovative TimeGPT model, worked on designing my own time-series forecasting model using the same architecture as the groundbreaking TimeGPT. By leveraging an encoder-decoder structure with multi-head attention, the model captures intricate patterns in historical data to make highly accurate predictions. Training it on my own dataset, I'm excited to explore the powerful potential of this architecture for time-series forecasting.

Traditional analysis methods such as ARIMA, ETS, MSTL, Theta, CES, machine learning models like XGBoost and LightGBM, and deep learning approaches have been standard tools for analysts. However, TimeGPT introduces a paradigm shift with its standout performance, efficiency, and simplicity.

**💡 Exciting Results:** Using this TimeGPT-inspired model, my training and testing results have significantly outperformed traditional models, demonstrating the superior predictive power of this approach!

![predictions_plot](https://github.com/user-attachments/assets/df0b84b6-31c2-42c8-86e1-e44a08be1060)

# Citation
Implementation of the deep learning architecture is based on this [paper](https://arxiv.org/abs/2310.03589):
```
@misc{garza2023timegpt1,
      title={TimeGPT-1},
      author={Azul Garza and Max Mergenthaler-Canseco},
      year={2023},
      eprint={2310.03589},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
