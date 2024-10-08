# Heat Pump Detection and Load Disaggregation in Smart Meter Data
This repo contains code for training models that detect heat pumps and disaggregate their load from aggregated Smart Meter Data (SMD). 
This includes minimal stages of preprocessing, model training and evaluation. Further, pretrained ```tensorflow``` models and scalers are included.


## Basic repo structure:
```
|-- app
|   |-- data_preparation
|   |-- models
|   |   |-- detection
|   |   |-- disaggregation
|-- data
|   |-- raw
|   |-- clean
|   |-- model_input
|-- legacy
|-- Pipfile
|-- Pipfile.lock
|-- configs.yaml
|-- secrets.yaml.template
```
### Notes on usage
* Pipenv is used for packaging and needs to be installed first, e.g., ```pip install pipenv``` 
* When code should be executed, make sure to set the working directory to the sources root of this repo: ```/app```
* There are multiple usecaeses that can/must be runned:
    * data preprocessing: transforms and cleans data, writes files to ```data/clean```
    * dataloaders: transforms data into numpy objects that can be used for model training, writes these files to ```data/model_input```
    * detection model training
    * disaggregation model training
  

### Notes on input data for heat pump detection
* Data is expected to be in the form of distinct pickeled ```pandas.DataFrame```s, corresponding to ids with a heat pump installed and
  ids where no heat pump is installed. 
* Each of these pickled ```pandas.DataFrame```s contains columns: ```id, timestamp, value_kwh```
* Each ```pandas.DataFrame``` contains one month of data or the usecase must be reconfigured
* A ```weather.p``` must be present containing at least columns ```timestamp, temp```, where ```timestamp``` 
  format and content should ideally match timestamps from consumption data. 

### Notes on input data for heat pump load disaggregation
* Data is expected to be in the form of distinct pickeled ```pandas.DataFrame```s, corresponding to ids where no heat pump is installed and
  ids where the only device is measured is a heat pump (heat pump load)
* Each of these pickled ```pandas.DataFrame```s contains columns: ```id, timestamp, value_kwh```
* Each ```pandas.DataFrame``` contains one month of data or the usecase must be reconfigured
* A ```weather.p``` must be present containing at least columns ```timestamp, temp```, where ```timestamp``` 
  format and content should ideally match timestamps from consumption data. 

## How to run
* Store raw files as described above in  ```data/raw```
* Run usecases in ```app/data_preprocessing/usecases```
* Select models, run dataloaders separately, train and evaluate models



> [!WARNING]
> This repository contains code written by a student. \
> The code may contain errors, bugs, or other issues. \
> **USE AT YOUR OWN RISK.** \
>If you find any errors or issues, please feel free to open an issue or submit a pull request.
