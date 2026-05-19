---

**Chest X-Ray Pneumonia Classifier**

Transfer Learning with DenseNet121 | PyTorch

---

A binary image classification project that detects Pneumonia from chest X-ray images using transfer learning on a pretrained DenseNet121 model. Built as part of a personal deep learning journey covering Tensors, Autograd, ANN, CNN, RNN, and LSTM from scratch in PyTorch.

---

**Dataset**

Chest X-Ray Images (Pneumonia) — available on Kaggle.

[https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1341 | 3875 | 5216 |
| Val (merged into train) | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

The original val set had only 16 images so it was merged with training data and re-split 90/10 for reliable validation metrics.

---

**Model Architecture**

Base model: DenseNet121 pretrained on ImageNet.

DenseNet uses dense connections where each layer receives feature maps from all previous layers. This makes it particularly well suited for medical imaging because subtle low-contrast features from early layers are preserved throughout the entire network.

```
DenseNet121 Backbone (partially frozen)
        ↓
  Linear(1024 → 512)
  BatchNorm1d(512)
  ReLU
  Dropout(0.4)
  Linear(512 → 128)
  ReLU
  Dropout(0.3)
  Linear(128 → 2)        ← binary classification
        ↓
  NORMAL  /  PNEUMONIA
```

Early layers of the backbone were frozen to preserve generic ImageNet features. The last two dense blocks were unfrozen for fine-tuning on the X-ray domain.

---

**Techniques Used**

*Handling Class Imbalance*

The dataset has roughly 3x more Pneumonia images than Normal. Two complementary methods were used:

- **WeightedRandomSampler** — oversamples the minority class so each training batch is balanced
- **Weighted CrossEntropyLoss** — penalizes mistakes on the minority class more during loss computation

*Data Augmentation (training only)*

- Random crop after resizing to 244×244
- Random horizontal flip (p=0.5)
- Random rotation ±10 degrees
- Color jitter on brightness and contrast

*Optimizer*

AdamW with differential learning rates — backbone at `lr=1e-5` and classifier head at `lr=1e-3`, with `weight_decay=1e-4`.

*Scheduler*

CosineAnnealingLR — smoothly decays learning rate over all epochs.

*Early Stopping*

Monitors validation loss with patience of 5. Best model weights saved and restored automatically.

---

**Training Results**

Training ran for 20 epochs. Early stopping triggered at epoch 20. Best weights restored from epoch 15.

| Epoch | Train Loss | Val Loss | Val Acc |
|-------|------------|----------|---------|
| 1 | 0.2788 | 0.3380 | 0.8394 |
| 5 | 0.2484 | 0.3065 | 0.8604 |
| 10 | 0.1721 | 0.2676 | 0.9006 |
| 15 ✅ | 0.2703 | 0.2367 | 0.9063 |
| 20 | 0.0622 | 0.2698 | 0.8509 |

---

**Test Set Evaluation**

```
              precision    recall  f1-score   support

      NORMAL     0.7322    0.9231    0.8166       234
   PNEUMONIA     0.9453    0.7974    0.8651       390

    accuracy                         0.8446       624
   macro avg     0.8387    0.8603    0.8409       624
weighted avg     0.8654    0.8446    0.8469       624

ROC-AUC Score: 0.9301
```

Confusion Matrix:

| | Predicted NORMAL | Predicted PNEUMONIA |
|---|---|---|
| Actual NORMAL | 216 (TN) | 18 (FP) |
| Actual PNEUMONIA | 79 (FN) | 311 (TP) |

---

**Threshold Optimization**

The default threshold of 0.5 gives Pneumonia recall of 0.80. Lowering the threshold to ~0.30–0.35 pushes Pneumonia recall above 0.95, which is more appropriate for a medical diagnosis context where missing a sick patient is far costlier than a false alarm.

ROC-AUC of 0.9301 confirms strong underlying discrimination ability regardless of threshold.

---

**Project Structure**

```
Pytorch-NNs-From-Scratch/
│
├── tensors/
│   └── tensors_in_pytorch.ipynb
│
├── autograd/
│   └── autograd.ipynb
│
├── dataset_loader/
│   └── datasetLoader_pytorch.ipynb
│
├── training_pipeline/
│   ├── Pytorch_NNmodule.ipynb
│   └── training_module_simpleNN.ipynb
│
├── ANN_MLP/
│   ├── ANN.ipynb
│   └── ann_fashion_mnist_pytorch_gpu.ipynb
│
├── CNN/
│   ├── CNNusingpytorch.ipynb
│   └── vgg16_model/
│       └── vgg16_finetuning.ipynb
│
├── RNN/
│   ├── RNN_pytorch.ipynb
│   └── Lstm_modelusingpytorch.ipynb
│
├── Projects/
│   └── ChestXRay_Classification/
│       └── ChestXRAY_DENSENET121.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

**Dependencies**

```
torch
torchvision
kagglehub
numpy
matplotlib
seaborn
scikit-learn
opencv-python
Pillow
```

---
