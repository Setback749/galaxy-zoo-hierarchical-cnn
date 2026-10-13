# Galaxy Zoo: Hierarchical vs End-to-End CNN

## Project Overview

This repository contains an AIML339 capstone project at Victoria University of Wellington investigating whether a simplified hierarchical convolutional neural network (CNN), inspired by the Galaxy Zoo decision tree, provides advantages over an end-to-end CNN for galaxy morphology classification.

The project compares predictive performance, training time, model complexity and the propagation of classification errors through hierarchical decisions.

The central research question is whether hierarchical classification offers benefits over an end-to-end approach, and how errors at earlier decision stages affect final predictions.

## Dataset

The project uses the [Galaxy Zoo Challenge dataset on Kaggle](https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge/data), the same source used by Jay Speidell in his Galaxy Zoo work.

A Kaggle account is required to download the dataset. Users may also need to accept the competition's terms before accessing the files.

The following files are required:

images_training_rev1.zip — Galaxy Zoo training images.

training_solutions_rev1.csv — volunteer classification response probabilities.

These files are not included in this GitHub repository due to their size and external distribution requirements. After downloading them, update the dataset paths in the notebook to point to their locations on your computer.

The original training dataset contains 61,578 galaxy images and corresponding volunteer response probabilities.

A reproducible random sample of 5,000 galaxies was selected using random state 42. After excluding seven samples whose dominant first-stage response was the star/artifact category, 4,993 galaxies remained.

The classification task uses three final classes:

- **Class 0:** Smooth
- **Class 1:** Featured, edge-on
- **Class 2:** Featured, not edge-on

The dataset was divided using stratified splitting into:

| Split | Images |
|---|---:|
| Training | 3,195 |
| Validation | 799 |
| Test | 999 |
| **Total** | **4,993** |

Images are loaded directly from the ZIP archive and resized to 64 × 64 pixels.

**Dataset files are not included in this repository.** Download `images_training_rev1.zip` and `training_solutions_rev1.csv` from Kaggle and configure their paths in the notebook.

## Models

### End-to-End CNN

A conventional CNN predicts one of the three final morphology classes directly.

### Hierarchical CNN

A CNN with a shared feature extractor and two classification heads follows a simplified decision hierarchy:

1. **Task 1:** Smooth or featured?
2. **Task 2:** If featured, edge-on or not edge-on?

Final predictions are produced using hard routing through the hierarchy. An incorrect Task 1 decision can therefore affect the final classification.

Both architectures use three convolutional blocks and a shared fully connected feature representation. Their trainable parameter counts are closely matched.

| Model | Trainable Parameters |
|---|---:|
| End-to-end CNN | 548,387 |
| Hierarchical CNN | 548,516 |

## Experimental Setup

Both models were trained using:

- PyTorch
- Image size: 64 × 64
- Batch size: 32
- Optimizer: Adam
- Learning rate: 0.001
- Loss: Cross-entropy, with separate task losses for the hierarchical model
- Maximum training duration: 10 epochs
- Model selection: Lowest validation loss

The repeated comparison used **30 matched random seeds (42–71)**. Both models were independently initialized and trained for each seed using the same fixed dataset splits.

Accuracy, macro-F1, training time and hierarchical error decomposition were recorded. Paired statistical tests were used to compare performance across seeds.

The held-out test set was reserved for a final evaluation using a representative seed selected from validation results.

## Results

### Repeated Validation Experiments

| Metric | End-to-End CNN | Hierarchical CNN |
|---|---:|---:|
| Mean accuracy | 74.82% | 74.21% |
| Mean macro-F1 | 0.7303 | 0.7240 |
| Mean training and validation time | 77.13 s | 78.48 s |

The end-to-end CNN achieved a mean accuracy advantage of approximately **0.62 percentage points**.

Paired t-tests across the 30 matched seeds found:

- **Accuracy:** p = 0.005353
- **Macro-F1:** p = 0.029982

These results indicate a small but statistically significant average advantage for the end-to-end CNN under the experimental setup.

After Holm correction for the two metric comparisons, the adjusted p-values were 0.0107 for accuracy and 0.0300 for macro-F1. Both remained below the significance threshold of 0.05.

### Hierarchical Error Propagation

Across the 30 validation runs, hierarchical classification errors were attributed to:

- **Task 1:** 4,913 errors (79.46%)
- **Task 2:** 1,270 errors (20.54%)

Mean hierarchical validation accuracy was **74.21%**. When ground-truth Task 1 routing was supplied for diagnostic evaluation, mean oracle-routing accuracy was **94.05%**.

Oracle routing uses ground-truth information and is not a deployable prediction method.

### Final Held-Out Test Evaluation

Using validation-selected seed 48 on 999 previously unseen test galaxies:

| Metric | End-to-End CNN | Hierarchical CNN |
|---|---:|---:|
| Accuracy | 73.57% | 72.47% |
| Macro-F1 | 0.7214 | 0.7157 |

The hierarchical model made 275 test classification errors, comprising 225 Task 1 errors and 50 Task 2 errors.

These single-seed test results are reported separately from the 30-seed validation statistics.

## Dependencies

- Python 3.11
- python -m pip install torch torchvision numpy pandas scikit-learn scipy matplotlib pillow jupyter

A CUDA-compatible GPU is recommended but is not required.

## Running the Project

1. Clone or download this repository.
2. Activate a suitable Python environment and install the dependencies.
3. Download the Galaxy Zoo training image archive and training solutions CSV from Kaggle.
4. Update the dataset paths in `AIML339 - Project Notebook.ipynb` to match your local environment.
5. Open the notebook in Jupyter or Positron.
6. Run the notebook cells in order to reproduce the preprocessing, model training, evaluation and analysis.

The repeated experiment loop saves completed seed results to `repeated_experiment_results.csv` and skips previously completed seeds when rerun.

**Note:** Reproducing all 30 seeds requires training both CNN architectures 30 times. Runtime depends on the available hardware.

**Reproducing the reported results:** The repository includes `repeated_experiment_results.csv`, which contains the saved results from the 30 matched-seed experiments. To reproduce the summary statistics and figures without retraining all models, run the notebook setup and analysis cells while skipping the repeated-training loop. To reproduce the full experiment from scratch, run the repeated-training loop using the documented seeds and settings. The final held-out test evaluation is a separate experiment using the representative seed selected from validation results.

## Repository Contents

- `AIML339 - Project Notebook.ipynb` — data preparation, model implementations, training, evaluation and analysis
- `repeated_experiment_results.csv` — results from 30 matched-seed validation experiments
- `final_test_results.csv` — final held-out test metrics
- `cnn_performance_boxplots.png` — validation accuracy and macro-F1 comparisons
- `cnn_paired_accuracy_differences.png` — paired validation accuracy differences
- `hierarchical_error_propagation.png` — hierarchical error decomposition
- `cnn_training_time_boxplot.png` — training-time comparison
- `cnn_learning_curves.png` — representative-seed training and validation loss curves
- `cnn_test_confusion_matrices.png` — side-by-side confusion matrices for the representative-seed held-out test evaluation, with per-class evaluation in the notebook

## Limitations and Future Work

The study uses a relatively small subset of the Galaxy Zoo dataset, a simplified three-class hierarchy and a fixed training configuration. The findings therefore should not be generalized to all hierarchical classification architectures.

Potential extensions include deeper Galaxy Zoo decision hierarchies, larger training datasets, alternative loss weighting and probabilistic routing to reduce error propagation.

## Data and Attribution

The galaxy images and volunteer-derived classification probabilities were obtained from the Galaxy Zoo Challenge dataset on Kaggle. Dataset files are not redistributed in this repository; users should obtain them directly from the original source and comply with the applicable access conditions.

The Galaxy Zoo decision hierarchy informed the simplified hierarchical classification design. [Jay Speidell's Galaxy Zoo Challenge project](https://jayspeidell.github.io/portfolio/project07-galazy-zoo/) was consulted as a reference for preprocessing and CNN-based image classification. No code was directly copied from that project.

OpenAI ChatGPT was used to assist with code development and debugging, experiment planning, statistical analysis, documentation and report drafting. The implementation, experiments and submitted work were reviewed and adapted by the project author.

This repository contains work produced for the AIML339 capstone project at Victoria University of Wellington.