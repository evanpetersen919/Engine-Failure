# Aircraft Engine RUL Prediction

Predictive maintenance project for aircraft engine Remaining Useful Life (RUL) estimation using the NASA C-MAPSS dataset.

## Project Overview

This project implements machine learning models to predict the Remaining Useful Life (RUL) of turbofan engines using sensor data. The goal is to enable proactive maintenance scheduling and prevent unexpected engine failures.

## Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**
- Training data: 100 engines with run-to-failure sensor readings
- Test data: Engines with unknown RUL to predict
- 21 sensor measurements per time cycle
- 3 operational settings per engine

Dataset files are located in the `CMaps/` directory:
- `train_FD001.txt` - Training data
- `test_FD001.txt` - Test data
- `RUL_FD001.txt` - Ground truth RUL values

## Project Structure

```
├── data_exploration.ipynb          # Exploratory data analysis
├── Step2_Data_Preprocessing.ipynb  # Data preprocessing pipeline
├── CMaps/                          # NASA C-MAPSS dataset
├── BUILD_IT_YOURSELF.md           # Project development guide
└── MILESTONES.md                  # Project milestones
```

## Setup

### Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow (for deep learning models)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Engine\ Failure

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## Usage

1. **Data Exploration**: Open `data_exploration.ipynb` to explore the dataset
2. **Data Preprocessing**: Run `Step2_Data_Preprocessing.ipynb` to prepare data for modeling
3. **Model Training**: (Coming soon)
4. **Evaluation**: (Coming soon)

## Project Status

- ✅ Data exploration completed
- 🚧 Data preprocessing in progress
- 📋 Model development planned
- 📋 Evaluation pipeline planned

## License

This project uses the publicly available NASA C-MAPSS dataset.

## References

- NASA Prognostics Data Repository
- C-MAPSS dataset documentation in `CMaps/Damage Propagation Modeling.pdf`
