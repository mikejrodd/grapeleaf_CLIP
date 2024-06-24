# Anomaly Detection in Vineyard Leaves Using CLIP and Isolation Forest

## Overview

This project aims to detect esca-infected vineyard leaves using an anomaly detection model that leverages CLIP for feature extraction and Isolation Forest for anomaly detection. The model's performance is evaluated based on its ability to correctly identify healthy and esca-infected leaves.

## Dataset

The dataset consists of 1805 images:
- 480 images of esca-infected leaves
- 1325 images of healthy leaves

## Methodology

1. **Feature Extraction**: CLIP (Contrastive Language–Image Pre-Training) is used to extract features from the leaf images.
2. **Anomaly Detection**: An Isolation Forest model is employed to detect anomalies (esca-infected leaves) in the feature space.

## Contamination Ratio

A contamination ratio of 0.1 is used for the Isolation Forest algorithm. Despite the actual proportion of esca-infected leaves being approximately 26.6%, this ratio is chosen to:
- Optimize the balance between detecting anomalies and minimizing false positives.
- Ensure the model generalizes well by being more discerning in flagging anomalies.

## Results

### Healthy Leaves
- **Precision**: 0.76
- **Recall**: 0.90

### Esca Leaves
- **Precision**: 0.42
- **Recall**: 0.19

### Overall Performance
- **Accuracy**: 0.71
- **Macro Average Precision**: 0.59
- **Macro Average Recall**: 0.55
- **Macro Average F1-score**: 0.54
- **Weighted Precision**: 0.66
- **Weighted Recall**: 0.71
- **Weighted F1-score**: 0.67

The model shows a good performance in identifying healthy leaves, with a high recall of 0.90, meaning it successfully identifies 90% of the healthy leaves. However, the precision and recall for esca-infected leaves are significantly lower, indicating that the model struggles to correctly identify esca-infected leaves.

## Conclusion

While the CLIP and Isolation Forest model is effective at identifying healthy leaves, its performance in detecting esca-infected leaves is suboptimal. The high recall for healthy leaves indicates reliability in recognizing normal patterns, but the low precision and recall for esca-infected leaves are significant drawbacks.

In comparison, a Convolutional Neural Network (CNN) model far outperformed the CLIP and Isolation Forest combination. CNNs are inherently more suited to image classification tasks and can more effectively differentiate between healthy and diseased leaves.

## Future Work

To improve the model's performance, future efforts should focus on:
- Integrating more advanced models like CNNs, which have demonstrated better capabilities in this domain.
- Improving the training dataset by including more diverse and representative samples of esca-infected leaves, potentially through synthetic data generation.
