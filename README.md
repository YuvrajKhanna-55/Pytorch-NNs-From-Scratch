Chest X-Ray Pneumonia Classifier :--
Transfer Learning with DenseNet121 | PyTorch

A binary image classification project that detects Pneumonia from chest X-ray images using transfer learning on a pretrained DenseNet121 model. Built as part of a personal deep learning learning journey covering tensors, autograd, ANN, CNN, RNN, and LSTM from scratch in PyTorch.

Dataset :--
Chest X-Ray Images (Pneumonia)- available on Kaggle.
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Model Architecture :--
Base model: DenseNet121 pretrained on ImageNet.
DenseNet uses dense connections where each layer receives feature maps from all previous layers. This makes it particularly well suited for medical imaging because subtle low-contrast features from early layers are preserved throughout the entire network.

DenseNet121 Backbone (partially frozen)
        ↓
  Linear(1024 → 512)
  BatchNorm1d(512)
  ReLU
  Dropout(0.4)
  Linear(512 → 128)
  ReLU
  Dropout(0.3)
  Linear(128 → 2) # binary classification
        ↓
  NORMAL  /  PNEUMONIA
Early layers of the backbone were frozen to preserve generic ImageNet features. The last two dense blocks were unfrozen for fine-tuning on the X-ray domain.

Techniques Used :--
1) Handling Class Imbalance
The dataset has roughly 3x more Pneumonia images than Normal. Two complementary methods were used to prevent the model from being biased toward the majority class.

2) WeightedRandomSampler — oversamples the minority class so each training batch is balanced
Weighted CrossEntropyLoss — penalizes mistakes on the minority class more during loss computation

3) Augmentation (training only)

Training Results :--
Training ran for 20 epochs before early stopping triggered at 20. Best weights from epoch 15 restored.
Train_loss=0.2703, Val_loss=0.2367, Val_acc=0.9063

Test Set Evaluation:--
              precision    recall  f1-score   support

      NORMAL     0.7322    0.9231    0.8166       234
   PNEUMONIA     0.9453    0.7974    0.8651       390

    accuracy                         0.8446       624
   macro avg     0.8387    0.8603    0.8409       624
weighted avg     0.8654    0.8446    0.8469       624

ROC-AUC Score: 0.9301

Project Structure:--
PyTorch-Learning/
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
│       ├── ChestXRAY_DENSENET121.ipynb
│
├── .gitignore
├── requirements.txt           
└── README.md                  


Dependencies:--
torch
torchvision
kagglehub
numpy
matplotlib
seaborn
scikit-learn
opencv-python
Pillow
