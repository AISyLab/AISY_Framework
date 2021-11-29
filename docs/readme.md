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

### Requirements

This framework requires ```tensorflow==2.0.0```, ```keras==2.1.6``` and ```python==3.7```.

All requires python packages are listed in ```requirements.txt```.

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
- Fully reproducible script   

### Example:

```python
import aisy_sca
from app import *
from custom.custom_models.neural_networks import *

aisy = aisy_sca.Aisy()
aisy.set_resources_root_folder(resources_root_folder)
aisy.set_database_root_folder(databases_root_folder)
aisy.set_datasets_root_folder(datasets_root_folder)
aisy.set_database_name("database_ascad.sqlite")
aisy.set_dataset(datasets_dict["ascad-variable.h5"])
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_batch_size(400)
aisy.set_epochs(20)
aisy.set_neural_network(mlp)
aisy.run()
```

### Running a script

After your AISY Framework project is set on your local machine, to run scripts that are placed inside ```scripts``` folder from 
command line or terminal, we recommend to write the following lines on your script (as the first lines of script code):

```python
import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))
```

However, if you prefer to run scripts from inside an IDE such as PyCharm, you don't have to set the above code in your scripts.

Alternatively, you can place your script files outside ```scripts``` folder (in your project root directory) and run them from there.

If you use our framework, please consider citing:

    @misc{cryptoeprint:2021:357,
      author = {Guilherme Perin and Lichao Wu and Stjepan Picek},
      title  = {{AISY - Deep Learning-based Framework for Side-Channel Analysis}},
      howpublished = {Cryptology ePrint Archive, Report 2021/357},
      note   = {\url{https://eprint.iacr.org/2021/357}},
      year   = {2021}
    }