<h3 align="center"> CaKE: Circuit-aware Editing Enables Generalizable Knowledge  Learners </h3>

## Table of Contents
- 🌟[Overview](#overview)
- 🔧[Installation](#installation)
- 🧐[Analyze](#analyze)
- 📚[Editing](#editing)
- 🌻[Acknowledgement](#acknowledgement)

---


## 🌟Overview
This work aims to improve the knowledge editing performance under the multi-hop reasoning settings.
## 🔧Installation

Build the environement:
```
conda create -n cake python=3.10
pip install -r requirements.txt
```

## 📚Analyze
Down load the wikidata for analysis from [HoppingTooLate](https://github.com/edenbiran/HoppingTooLate/blob/main/datasets/two_hop.csv) into `Analysis/datasets`.

- Run `evaluate_dataset.py` to filter the data.
- Run `generate_entity_description.py` to get the entity patch and relation patch.
- Run `patch_activations.py` to do back-patching and cross-patching.
- Run `analysis.ipynb`

## 🧐Editing
Just run the following commond:
```
run.sh
```

## 🌻Acknowledgement

We thank for the project of [EasyEdit](https://github.com/zjunlp/EasyEdit), [HoppingTooLate](https://github.com/edenbiran/HoppingTooLate).
The code in this work is built on top of these projects' codes.