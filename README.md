# SimpleDL

**SimpleDL** is a lightweight deep learning framework built from scratch using NumPy. It's designed for **educational purposes** to help understand how neural networks work under the hood.

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- **Sequential API** - Build neural networks layer by layer
- **Linear Layer** - Fully connected layer with He initialization
- **Activation Functions** - ReLU, Sigmoid, Tanh, Softmax
- **Regularization** - Dropout layer to prevent overfitting
- **Loss Functions** - MSE, Binary Cross-Entropy, Categorical Cross-Entropy
- **Optimizers** - SGD and Adam (with bias correction)
- **Utilities** - One-Hot Encoding
- **Pure NumPy** - No external dependencies except NumPy

## 📦 Installation

```bash
pip install numpy
```
## Then clone the repository:
```bash
git clone https://github.com/NxFi3/SimpleDL.git
cd SimpleDL
```
## 🚀 Quick Start
- XOR Problem Example
```python
#import module

from SimpleDL import activations ,losses
from SimpleDL import neuralnetwork as nn
from SimpleDL import optimizer as optim
from SimpleDL import utils
import numpy as np

#create Simple Model and Optimizer

model = nn.Sequential(nn.Linear(2,16),activations.ReLU(),nn.Linear(16,2))
optimizer = optim.Adam(model,losses.CrossEntropyLoss(),learning_rate=0.003)

#Create XOR dataset

#encoder for CrossEntropyLoss
encoder = utils.OneHotEncoding(2)

X = np.array([[1,0],[0,1],[1,1],[0,0]])

Y = encoder.encode(np.array([1,1,0,0]))

#Train Model

epoches=100
for ep in range(epoches):
    for x ,y in zip(X,Y):
        pred = model(x)
        loss = optimizer.step(pred,y)
    if ep%10==0:
        print(f"loss: {loss:4f}")

# Test Result
print('==== Test Result ====')
for x in X:
    pred = model(x)
    print(f"Input : {x} ,Model Predict : {np.argmax(pred)}")
```
## 📁 Project Structure
```text
SimpleDL/
├── SimpleDL/  
│   ├── activations.py
│   ├── losses.py
│   ├── neuralnetwork.py
│   ├── optimizer.py
│   └── utils.py
├── requirements.txt
├── example.py
└── README.md
```

## 📖 API Reference

### Layers

| Class | Description |
|-------|-------------|
| `Linear(inputs, outputs)` | Fully connected layer |
| `Sequential(*layers)` | Container for stacking layers |

### Activation Functions

| Class | Formula | Range |
|-------|---------|-------|
| `ReLU()` | max(0, x) | [0, ∞) |
| `Sigmoid()` | 1/(1+e^(-x)) | (0, 1) |
| `Tanh()` | (e^x-e^(-x))/(e^x+e^(-x)) | (-1, 1) |
| `Softmax()` | e^x_i / Σe^x_j | (0, 1), sum=1 |
| `Dropout(p)` | Randomly zeros with probability p | Regularization |

### Loss Functions

| Class | Use Case |
|-------|----------|
| `MSELoss()` | Regression |
| `BinaryCrossEntropyLoss()` | Binary classification |
| `CategoricalCrossEntropyLoss()` | Multi-class classification |
| `CrossEntropyLoss()` | Auto-detects binary/multi-class |

### Optimizers

| Class | Learning Rate | Best For |
|-------|---------------|----------|
| `SGD(model, loss_fn, lr)` | 0.001 - 0.1 | Simple problems |
| `Adam(model, loss_fn, lr=0.001)` | 0.0001 - 0.01 | Most problems (recommended) |

## 💡 Examples

### Binary Classification (Apple vs Orange)

```python
model = nn.Sequential(
    nn.Linear(12288, 64),  # 64x64 image → 12288 features
    activations.ReLU(),
    nn.Linear(64, 1),
    activations.Sigmoid()
)

optimizer = optim.Adam(model, losses.BinaryCrossEntropyLoss(), lr=0.001)
```

### Regression

```python
model = nn.Sequential(
    nn.Linear(5, 10),
    activations.Tanh(),
    nn.Linear(10, 1)
)

optimizer = optim.SGD(model, losses.MSELoss(), lr=0.01)
```

## 🔧 Requirements

- Python ≥ 3.8
- NumPy

## 🧪 Testing

Run the XOR example:

```bash
python example.py
```

Expected output:

```
loss: 0.617449
loss: 0.589605
loss: 0.566769
loss: 0.556935
loss: 0.545078
loss: 0.530720
==== Test Result ====
Input : [1 0] ,Model Predict : 1
Input : [0 1] ,Model Predict : 1
Input : [1 1] ,Model Predict : 0
Input : [0 0] ,Model Predict : 0
```

## 🤝 Contributing

Feel free to open issues or submit pull requests. Suggestions for improvements are welcome!

## 📄 License

MIT License - feel free to use this project for learning and teaching.

## 👨‍💻 Author

**NxFi3**

- GitHub: [@NxFi3](https://github.com/NxFi3)

## ⭐ Show Your Support

If you found this helpful, please give it a ⭐ on GitHub!

### `requirements.txt`
```txt
numpy>=1.24.0
```
