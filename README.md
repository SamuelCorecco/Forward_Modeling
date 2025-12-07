# Solar Stokes Forward Modeling: Machine Learning Methods Analysis

This project implements and compares several Machine Learning and Deep Learning architectures for Forward Modeling in solar physics.  
The goal is to emulate the radiative transfer equation by predicting the Stokes profiles ($I, Q, U, V$) from atmospheric physical parameters (Temperature, Magnetic Field Strength, Inclination, Azimuth).

The project benchmarks several state-of-the-art regression methods (XGBoost, PCA/F-PCA) and reproduces the study by **Carroll et al. (2008)**, which proposed a fast method for Stokes profile synthesis, comparing them against our newly developed Deep Learning approach based on a **Multi-ResNet architecture with adaptive physics-based normalization**.

## Citation

    @article{Carroll_2008,
       title={A fast method for Stokes profile synthesis: Radiative transfer modeling for ZDI and Stokes profile inversion},
       volume={488},
       ISSN={1432-0746},
       url={http://dx.doi.org/10.1051/0004-6361:200809981},
       DOI={10.1051/0004-6361:200809981},
       number={2},
       journal={Astronomy & Astrophysics},
       publisher={EDP Sciences},
       author={Carroll, T. A. and Kopf, M. and Strassmeier, K. G.},
       year={2008},
       month={jul},
       pages={781–793}
    }

---

# Project Structure

```text
solar_project/
├── config/               
│   └── experiments/
│       ├── multi_resnet.yaml   
│       ├── fpca_xgboost.yaml    
│       ├── xgboost_full.yaml    
│       └── paper2008_mlp.yaml   
│
├── data/                    
│   └── atmoFe.npz         
│
├── results/                 
│
├── src/                     
│   ├── data/                
│   ├── features/            
│   ├── models/              
│   │   ├── components/      
│   │   └── wrappers.py      
│   └── model_builder.py     
│
├── run_experiment.py        
├── summary_results.py      
├── visualize_results.py     
└── requirements.txt   
```

# Quick Start

## 1. Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

## 1. Dataset Preparation
Ensure the dataset file is placed correctly:
- **Path**: data/atmoFe.npz

## 3. Run an Experiment
To launch model training using a chosen configuration:
```bash
python run_experiment.py --config config/experiments/resnet_sobolev.yaml
```
After execution, the script automatically saves:
- **metrics.csv**: RMSE, CCC, Peak Error, Derivative Correlation
- **performance.csv**: training/inference timings
- **model.pkl**: trained model
Inside:
```text
results/<experiment_name>/
```

## 4. Compare Results
Generate a comparative table (with color highlighting) of all experiments:
```bash
python summary_results.py
```


## 5. Visualize Stokes Profiles
To generate plots comparing real vs predicted profiles:
```bash
python visualize_results.py --config config/experiments/multi_resnet.yaml --indices 0 42 100 150
```
This generates the file:
```text
reconstruction_examples.png
```
inside the corresponding experiment results directory.
---

# Evaluation Metrics

The project evaluates models using both **statistical** and **physics-informed** metrics to ensure that the predicted Stokes profiles are not only numerically accurate but also physically meaningful.

### **RMSE — Root Mean Squared Error**  
Measures the average squared difference between predicted and true Stokes profiles.  
Useful for general regression quality.

### **CCC — Concordance Correlation Coefficient**  
Quantifies how well predictions match the ideal relation \( y = x \).  
Penalizes both **bias** and **scale mismatch**, making it more informative than simple correlation.

### **Peak Error (%)**  
Measures the relative error on the maximum amplitude of each Stokes profile.  
This is critical for solar physics because the amplitude of polarized lines correlates strongly with the **magnetic field strength**.

### **Derivative Correlation**  
Computes the correlation between the **first derivatives** of the predicted and true profiles.  
This captures the **shape**, **lobe structure**, and **asymmetries**, independent of absolute intensity.

---

# Results

Below are the main evaluation tables for all experiments executed.  
Bold entries represent the best performance per column.

## **RMSE**

| Experiment     |       I |       Q |       U |       V |    MEAN |
|----------------|---------|---------|---------|---------|---------|
| pca_xgboost    | 0.0102  | 0.00614 | 0.00605 | 0.00768 | 0.00752 |
| fpca_xgboost   | 0.01029 | 0.00649 | 0.00613 | 0.00769 | 0.00765 |
| paper2008      | 0.01059 | 0.00402 | 0.00407 | 0.0073  | 0.0065  |
| **resnet_sobolev** | **0.00091** | **0.0006** | **0.00061** | **0.00074** | **0.00072** |
| xgboost_full   | 0.00334 | 0.00751 | 0.00745 | 0.00707 | 0.00634 |

## **CCC**

| Experiment     |       I |       Q |       U |       V |    MEAN |
|----------------|---------|---------|---------|---------|---------|
| pca_xgboost    | 0.99462 | 0.44864 | 0.43267 | 0.82144 | 0.67434 |
| fpca_xgboost   | 0.9946  | 0.4488  | 0.43361 | 0.82112 | 0.67453 |
| paper2008      | 0.99451 | 0.45219 | 0.4373  | 0.82153 | 0.67638 |
| **resnet_sobolev** | **0.99607** | **0.46201** | **0.44694** | **0.8293**  | **0.68358** |
| xgboost_full   | 0.99589 | 0.44569 | 0.42913 | 0.82348 | 0.67355 |

## **Peak Error (%)**

| Experiment     |       I |        Q |       U |       V |     MEAN |
|----------------|---------|----------|---------|---------|----------|
| pca_xgboost    | 0.21429 | 14.5506  | 14.9046 | 4.87502 |  8.63614 |
| fpca_xgboost   | 0.22999 | 16.0883  | 15.8843 | 5.09964 |  9.32558 |
| paper2008      | 0.23285 | 10.7106  | 11.2383 | 5.61253 |  6.94858 |
| **resnet_sobolev** | **0.01627** | **2.52634** | **2.8788** | **0.87096** | **1.57309** |
| xgboost_full   | 0.00369 | 19.299   | 19.408  | 6.9108  | 11.4054  |

## **Derivative Correlation**

| Experiment     |       I |       Q |       U |       V |    MEAN |
|----------------|---------|---------|---------|---------|---------|
| pca_xgboost    | 0.97791 | 0.68952 | 0.67731 | 0.87014 | 0.80372 |
| fpca_xgboost   | 0.97758 | 0.68766 | 0.67043 | 0.86782 | 0.80087 |
| paper2008      | 0.97738 | 0.71738 | 0.70438 | 0.87002 | 0.81729 |
| **resnet_sobolev** | **0.99997** | **0.80298** | **0.79482** | **0.90975** | **0.87688** |
| xgboost_full   | 0.99952 | 0.72421 | 0.71366 | 0.9045  | 0.83547 |

## **Performance**

| Experiment     |   train_time_s |   throughput_ms |   latency_ms |
|----------------|----------------|-----------------|--------------|
| pca_xgboost    | 19.3983        | 0.07033         | 1303.35      |
| fpca_xgboost   | 18.3872        | 0.01924         | 112.473      |
| paper2008      | 259.768        | 0.00638         | 0.42656      |
| **resnet_sobolev** | 4158.56        | 0.02435         | 9.92212       |
| xgboost_full   | 105.574        | 0.33767         | 7361.5       |

- **throughput_ms**  
  This is the average time (in milliseconds) required for the model to predict one sample when processing the entire test set as a batch.  
  It is computed as: throughput_ms = (total_prediction_time / number_of_samples) * 1000
- **latency_ms**  This is the time needed to predict a single sample, measured by repeatedly calling the model on just one input: latency_ms = (time_for_50_single_predictions / 50) * 1000

---