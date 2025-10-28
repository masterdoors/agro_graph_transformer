# Source code and data 
for Yulia Otmakhova, Dmitry Devyatkin, and He Zhou "Hybrid supply chain model for wheat market"

## Structure
### Folders:
- /trade_data contains original international wheat trade data files from Comtrade
- /prod_data contains original  wheat production data files from FAOSTAT
- /layers - modified graph transformer layers from https://github.com/hyunwoongko/transformer
- /test_outputs - contains test results

### Notebooks:
- create_dataset.ipynb - creates dataset from the original FAOSTAT and Comtrade data
- arima_and_test_distribution.ipynb - train ARIMA and check the target distribution
- experiments.ipynb - experiments with RNNs, and with proposed Transformer-based models
- plot.ipynb - draw diagrams for the paper

