# AISY - Deep Learning-based Framework for Side Channel Analysis

Welcome to the first deep learning-based side-channel analysis framework.
This AISY Framework was developed by AISYLab from TU Delft.

Contributors: Guilherme Perin, Lichao Wu and Stjepan Picek.

### Installation

```
git clone https://github.com/AISyLab/AISY_Framework.git
cd AISY_framework
pip install -r requirements.txt
```

To start the webapp:

```
flask run
```

### Documentation

See our documentation page: https://aisylab.github.io/AISY_docs/

### Main Features

- SCA Metrics
- Gradient Visualization
- Data Augmentation 
- Grid Search
- Random Search
- Early Stopping
- Ensemble
- Custom Metrics
- Custom Callbacks
- Confusion Matrix
- Easy Neural Network Definitions
- Data Augmentation
- GUI - plots, tables
- Automatically generate scripts    

### Example:

```python
from custom.custom_models.neural_networks import *
from aisy.sca_deep_learning_aes import AisyAes

aisy = AisyAes()
aisy.set_database_name("database_ascad.sqlite")
aisy.set_dataset("ascad-variable.h5")
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_batch_size(400)
aisy.set_epochs(50)
aisy.set_neural_network(mlp)

aisy.run()
```

If you use our framework, please consider citing:

    @misc{AISY_Framework,
      author = {Guilherme Perin and Lichao Wu and Stjepan Picek},
      title  = {{AISY - Deep Learning-based Framework for Side-Channel Analysis}},
      howpublished = {AISyLab repository},
      note   = {{\url{https://github.com/AISyLab/AISY_Framework}}},
      year   = {2021}
    }