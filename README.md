# Pirelli Tech Challenge - Feature Engineering 

## Introduction

This project aims at generate the dataset of features for a predictive system that, based on a certain set of metrics, will predict when to stop a machine for maintenance actions. In particular, the tracked machines are involved in a three phases cooking process, which is organized as *kitchens* that have a set of *machines*, each of those is associated with a specific *cooking phase*. 

In the first version, data regards a specific combination of those, namely kitchen k1, machine m1 (associated to cooking phase 1),
and arepa type a1. The following datasets are available for the analysis:
* Cooking metrics: a table containing cooking metrics retrieved by each machine associated to phase 1 for each cooked batch.
* Batch registry: a table containing the main information of each produced batch.
* Faulty intervals: a table containing for each machine associated to phase 1 the intervals in which metrics data are note reliable.

The program is expected to receive as input start and end times and produce a dataset with the hourly averaged metrics for kitchen k1 and machine m1 associated to arepa type a1 in the specified time interval. Keep in mind that faulty data must be filtered out.

## Project structure

* **source/**: contains source datasets
* **outputs/**: contains output dataset with subfolder structure as **\<phase\>/\<start_date\>_\<end_date\>/**. The output file name is built based on kitchen, machine IDs and the areapa type. For example, giving the start date and end date equal to **2020-11-01T00:23:34 and **2020-11-01T01:23:34 respectively, the output dataset will be located at **outputs/phase1/20201101T002334_20201101T022334/k1_m1_a1.csv**
* **feature_eng.py**: main script
* **unit_tests.py**: Unit test script

## How to use
### Dependencies ###

* You need at least [Python 3.x with pip](https://www.python.org/downloads/) 
* pandas

    ```bash
   pip install pandas
   ```
## Run the main script ###


```bash
python feature_eng.py <start_date> <end_date>
```

Both start and end date must be provided in the format **YYYYMM-
DDTHH:MM:SS**

## Run unit tests ###


```bash
python unit_tests.py
```

## Roadmap

In the next release, training must be scaled to consider all the machines in all the kitchens associated to phase 1, but reduced to a list of arepa types given in a text file. Thus, we need to apply the following refactoring tasks:

* provide the location of the arepa list file as user input
* Modify the *build_features()* function in order to take 


## Authors
* Mauro Losciale(https://it.linkedin.com/in/maurolosciale)
