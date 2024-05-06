# Project Name

![Banner Image](path/to/banner-image.png)

## Introduction / Summary

This Github is the companion to the paper 'Evaluating Language Model Language Traits'
## Repository Structure

This repository is organized into several main directories:

### `/HHH`

This subdirectory contains all code and data related to the generation of the Helpful Harmless (HHH) dataset.

- **Description**: 
- **Contents**:
  - `src`: This folder contains all the source code for the project. It includes scripts, modules, and other code files that are directly used to run the project.
	  - `/modules`: Subdirectory for project-specific modules.
	  - `/utilities`: Helper scripts and utility functions
- `/experiments`
	- run_experiments.py - script relating to testing the dataset and model combinations for few shot and chain of thought prompting
 

- `/data` - **Description**: This directory stores project-related data files. Depending on the size of the data, this might only include sample data or metadata with links to the full datasets stored externally.
- **Contents**:
  -  dataset: the raw dataset scenarios generated by GPT-4
  - dataset_with_adapt: the HH dataset with the adaptation sentance added by GPT-4
  -processed: the results from the dataset processed with normal prompting call 
  -processed_chainofthought: the results from the dataset procesed with COT prompting call
  -Process_fewshot_X: the results from the dataset processed with few shot prompting with X examples 
  
 

## Getting Started

Provide instructions on how to set up, configure, and run the project locally. This might include steps to install software, configure environments, and execute scripts:

```bash
git clone https://github.com/yourusername/projectname.git
cd projectname
pip install -r requirements.txt
python src/main.py



