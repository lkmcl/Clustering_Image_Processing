# Low-Rank Image Classification with K-Means and SVM

**Contributors:** Logan McLaurin, Michael Brown, Myles Gregor, Bryant Willoughby 

This repository contains the full set of files from a collaborative project completed during Spring 2024 for Mathematical Foundations of Data Science (MA 326) at North Carolina State University.

## Personal Contributions

### Development of all SVM Algorithms:
- Includes the following scripts: `SVMComplex.py`, `SVMCommon.py`, `SVMSimple.py`
- Designed, tested, and executed all scripts related to the Support Vector Machine (SVM) classification algorithm, leveraging a linear kernel to optimize model performance on both simple (UCI handwritten digits) and complex (CIFAR-10) datasets
- Conducted parameter tuning to balance training accuracy and generalization, achieving near-perfect accuracy on simple datasets and identifying overfitting trends on complex data

### Data Prep
- Sourced and preprocessed the CIFAR-10 dataset, ensuring compatibility with project requirements by adapting its RGB format for dimensionality reduction and classification
- Generated training and testing partitions for the CIFAR-10 dataset using custom Python functions, maintaining a balanced distribution across all 10 classes

### Results, Analysis, and Documentation
- Drafted the Results section of the project report, interpreting statistical outputs and visualizations to communicate key findings on the effects of low-rank approximation on classification accuracy
- Co-authored the Conclusions section, synthesizing project insights to highlight trade-offs between computational efficiency and model performance

### Technical Peer Review
- Conducted peer reviews of all scripts, including those for low-rank approximation (LRA) and K-Means clustering, to ensure code quality, logic accuracy, and alignment with project objectives
- Provided constructive feedback on methodology, particularly in evaluating the appropriateness of rank thresholds and the clustering algorithm's initialization techniques

---
## Overview
This project investigates the effects of low-rank approximation on the performance of unsupervised and supervised image classification algorithms: K-Means Clustering and **Support Vector Machines (SVM). By reducing image dimensionality through singular value decomposition (SVD), we assess how classification accuracy changes on both simple (UCI handwritten digits) and complex (CIFAR-10) datasets. The analysis demonstrates how intrinsic image dimensions affect model performance, highlighting the trade-offs between computational efficiency and accuracy

---
## Data Description
### UCI Handwritten Digits Dataset
- **Type:** Grayscale, 8x8 pixel images
- **Classes:** 10 (digits 0–9)
- **Dataset Size:** 1,797 images
- **Purpose:** Simple image classification benchmark

### CIFAR-10 Dataset
- **Type:** RGB, 32x32 pixel color images
- **Classes:** 10 (airplane, automobile, bird, etc.)
- **Dataset Size:** 2,000 randomly selected images
- **Purpose:** Complex image classification challenge

---
## Methods
### Low-Rank Approximation
- Used Singular Value Decomposition (SVD) to reduce image dimensionality
- Generated rank-reduced approximations for ranks ranging from 1, 2, 3, 4, and then multiples of 5% up to 100% of original image size

### K-Means Clustering
- Unsupervised learning algorithm for clustering image data
- Utilized initialization techniques and centroid updates to achieve local optima for cluster assignments

### Support Vector Machines (SVM)
- Supervised learning algorithm for binary and multi-class classification
- Employed a linear kernel for efficient computation and scalability

---
## Results
- K-Means Clustering:
  - Achieved ~75% accuracy on the UCI dataset but struggled (~25% accuracy) on the CIFAR-10 dataset, even at full rank
  - Accuracy plateaued beyond rank **2 or 3**, highlighting low intrinsic dimensions of the data

- SVM Classification:
  - Near-perfect classification on the UCI dataset (~98% accuracy).
  - High training accuracy on CIFAR-10 (~100%) but limited test accuracy (~30%), suggesting overfitting to the training data.

## Scripts and Functions
### Low-Rank Approximation (LRA.py)
- `LRA(img, rank)`: Computes rank-reduced RGB image approximations.
- `LRA_grayscale(img, rank)`: Computes rank-reduced grayscale image approximations.

### SVM Functions (SVMCommon.py, SVMComplex.py, SVMSimple.py)
- `partition(data, target, p)`: Splits data into training and testing sets.
- `svc_analysis(train_data, train_target, test_data, test_target)`: Calculates SVM training and testing accuracy.

### K-Means Functions (kmeansCommon.py, kmeansComplicated.py, kmeansSimple.py)
- `partition(data, target, p)`: Splits data into training and testing sets.
- `randomInit(vectorLength)`: Initializes random centroids for K-Means clustering.
- `KMeansClusters(initializationTechnique, train_data)`: Clusters training data using the K-Means algorithm.

---

## Key Findings
- Intrinsic dimensionality plays a critical role in image classification performance.
- SVD-based low-rank approximations retain sufficient data for accurate classification, even at reduced dimensions.
- SVM consistently outperformed K-Means across all datasets, emphasizing the benefits of supervised learning for complex classification tasks.
