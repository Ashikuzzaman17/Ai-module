#  Bone Fracture Detection System (Computer Vision Pipeline)

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.18.0-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An automated deep learning pipeline engineered to detect bone fractures from digital X-ray imagery. The model processes input radiographs, optimizes feature clarity using classic computer vision enhancement techniques, and delivers binary classifications backed by diagnostic confidence intervals.

---

##  Key Pipeline Features

- **Advanced Pre-processing:** Implements Contrast Limited Adaptive Histogram Equalization (**CLAHE**) and Gaussian blurring to accentuate structural fractures often hidden in raw scans.
- **Deep Feature Extraction:** Built upon an **EfficientNetB0** architecture pre-trained on ImageNet as an immutable feature map backbone, matched with a regularized head.
- **Explainable Metrics Evaluation:** Beyond simple precision/recall, the engine evaluates model certainty using **95% Bootstrap Confidence Intervals** and Cohen's Kappa indices.
- **Performance Timer Wrapper:** Embeds an isolated `Timer` class to compute computational overhead and training bottlenecks across target epochs.

---

##  Tech Stack

- **Core Engine:** TensorFlow 2.18.0, Keras
- **Image Processing & Computer Vision:** OpenCV (`cv2`), Scikit-Image (`skimage`)
- **Evaluation & Diagnostics:** Scikit-Learn, Statsmodels
- **Data & Visual Analytics:** NumPy, Pandas, Matplotlib, Seaborn

---

##  System Metrics Summary

The platform operates as a binary classifier evaluated on individual image splits:

| Dataset Split | Precision (Normal) | Precision (Fracture) | Total Accuracy | 95% Confidence Interval |
| :--- | :--- | :--- | :--- | :--- |
| **Training Split** | 0.00 | 0.94 | 94.09% | — |
| **Validation Split** | 0.00 | 0.94 | 94.04% | — |
| **Testing Split** | 0.00 | 0.94 | 0.9406% | **[90.60% – 95.95%]** |

*Note: The highly skewed dataset metrics reflect a structural class imbalance (2,000 fracture vs 127 normal cases), highlighting the pipeline's emphasis on true recall tracking.*

---

##  Project Setup & Installation

Ensure you are utilizing Python 3.11+. Clone the repository, create a virtual workspace, and deploy dependencies:

```bash
# Clone the repository
git clone [https://github.com/Ashikuzzaman17/Ai-module.git](https://github.com/Ashikuzzaman17/Ai-module.git)
cd Ai-module

# Build and activate environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt