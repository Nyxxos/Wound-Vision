# Wound Vision

Wound Vision is an educational computer vision project using ResNet-18 and transfer learning to classify wound images.

The current model classifies five classes:

- Abrasions
- Bruises
- Burns
- Cut
- Normal

The project currently covers the full image classification pipeline from preparing the dataset and training the model to evaluation, inference and Grad-CAM analysis.

## Current Results

The final model was tested on 126 held-out images.

- Accuracy: **92.06%**
- Macro Precision: **92.70%**
- Macro Recall: **91.47%**
- Macro F1: **91.81%**
- Correct predictions: **116 / 126**

## Model

The current model uses a pretrained ResNet-18.

For V1, the ResNet backbone is frozen and only the final five-class classifier is trained. I used transfer learning because the dataset is relatively small and wanted to first build a stable baseline before trying fine-tuning.

## Project Flow

```text
dataset.py
    ↓
model.py
    ↓
train.py
    ↓
evaluate.py
    ↓
inference.py
    ↓
gradcam.py
    ↓
gradcam_analysis.py
    ↓
Flask web app
```

## V1 Progress

Completed:

- Dataset preparation
- ResNet-18 transfer learning model
- Training and validation
- Held-out test evaluation
- Confusion matrix and class metrics
- Single-image inference
- Top-3 predictions
- Grad-CAM
- Grad-CAM analysis

Still to do:

- Flask web app
- Image upload
- Prediction result page
- Top-3 display
- Grad-CAM display
- Input validation
- Deployment
- Final screenshots

## Documentation

More detailed notes explaining the decisions behind each part of the project are available in the `docs` folder.

```text
docs/
├── README.txt
├── dataset.txt
├── model.txt
├── train.txt
├── evaluate.txt
├── inference.txt
├── gradcam.txt
├── gradcam_analysis.txt
└── project_overview.txt
```

These notes explain what each part does, why I chose the current approach, what can be changed and what I may test in a later version.

## Disclaimer

Wound Vision is an educational machine learning project and is not intended for medical diagnosis or treatment decisions.
