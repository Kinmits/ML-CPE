# LAB 06: Neural Network (NN) and Its Applications

## Objective
- Apply a Multi-Layer Perceptron (MLP) Neural Network to classify Fantasy Premier League (`fpl.csv`) player positions (`GKP`, `DEF`, `MID`, `FWD`).
- Compare the model performance across different numbers of **epochs** (50, 100, 200, 500) and **Neural Network configurations** (1 Layer: 32 units, 1 Layer: 64 units, and 2 Layers: 64x32 units).

## Directory Structure
```text
LAB-06/
│
├── fpl.csv                 # FPL Player Dataset
├── README.md               # Project documentation
│
└── classification/         # Modular Package
    ├── __init__.py
    ├── data_loader.py       # Load dataset
    ├── preprocessing.py     # Feature selection & StandardScaler
    ├── split_data.py        # Train/Test Split (80/20)
    ├── nn_model.py          # MLPClassifier training across configs & epochs
    ├── evaluate.py          # Metrics evaluation & chart generation
    ├── main.py              # Main execution script
    ├── test_nn.py           # Quick integration test
    └── output/              # Generated plots and visualizations