# FFORMA-ESRNN
State of the art time series forecasting method that has the FFORMA ensemble learn from the ESRNN hybrid model and others.

 <p align="center">
    <img src="resources/results.PNG" alt="alternate text">
 </p>


## To install: <br>

conda env create -f environment.yml
conda activate fformaProj

## To run: <br>

```bash
python main.py
```

## Check GPU: <br>

```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```