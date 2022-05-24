# [FiNER: Financial Numeric Entity Recognition for XBRL Tagging](https://arxiv.org/abs/2203.06482)

Publicly traded companies are required to submit periodic reports with eXtensive Business Reporting Language (XBRL) word-level tags. Manually tagging the reports is tedious and costly. We, therefore, introduce XBRL tagging as a new entity extraction task for the financial domain and release FiNER-139, a dataset of 1.1M sentences with gold XBRL tags. Unlike typical entity extraction datasets, FiNER-139 uses a much larger label set of 139 entity types. Most annotated tokens are numeric, with the correct tag per token depending mostly on context, rather than the token itself. We show that subword fragmentation of numeric expressions harms BERT's performance, allowing word-level BILSTMs to perform better. To improve BERT's performance, we propose two simple and effective solutions that replace numeric expressions with pseudo-tokens reflecting original token shapes and numeric magnitudes. We also experiment with FIN-BERT, an existing BERT model for the financial domain, and release our own BERT (SEC-BERT), pre-trained on financial filings, which performs best. Through data and error analysis, we finally identify possible limitations to inspire future work on XBRL tagging.

---

## Citation Information

```text
@inproceedings{loukas-etal-2022-finer,
    title = {FiNER: Financial Numeric Entity Recognition for XBRL Tagging},
    author = {Loukas, Lefteris and
      Fergadiotis, Manos and
      Chalkidis, Ilias and
      Spyropoulou, Eirini and
      Malakasiotis, Prodromos and
      Androutsopoulos, Ion and
      Paliouras George},
    booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022)},
    publisher = {Association for Computational Linguistics},
    location = {Dublin, Republic of Ireland},
    year = {2022},
    url = {https://arxiv.org/abs/2203.06482}
}
```

---

## Table of Contents
* [Dataset and Supported Task](#dataset-and-supported-task)
* [Dataset Repository](#dataset-repository)
* [Models Repository](models-repository)
* [Install Python and Project Requirements](#install-python-and-project-requirements)
* [Running an Experiment](#running-an-experiment)
* [Setting up the experiment's parameters](#setting-up-the-experiment's-parameters)

___

## Dataset and Supported Task

FiNER-139 is comprised of 1.1M sentences annotated with eXtensive Business Reporting Language (XBRL) tags extracted from annual and quarterly reports of publicly-traded companies in the US. Unlike other entity extraction tasks, like named entity recognition (NER) or contract element extraction, which typically require identifying entities of a small set of common types (e.g., persons, organizations), FiNER-139 uses a much larger label set of 139 entity types. Another important difference from typical entity extraction is that FiNER focuses on numeric tokens, with the correct tag depending mostly on context, not the token itself.

To promote transparency among shareholders and potential investors, publicly traded companies are required to file periodic financial reports annotated with tags from the eXtensive Business Reporting Language (XBRL), an XML-based language, to facilitate the processing of financial information. However, manually tagging reports with XBRL tags is tedious and resource-intensive. We, therefore, introduce XBRL tagging as a new entity extraction task for the financial domain and study how financial reports can be automatically enriched with XBRL tags. To facilitate research towards automated XBRL tagging we release FiNER-139.

## Dataset Repository

[FiNER-139](https://huggingface.co/datasets/nlpaueb/finer-139) is available at HuggingFace Datasets and you can load it using the following:

```python
import datasets

finer = datasets.load_dataset("nlpaueb/finer-139")
```

Note: You don't need to download or install any dataset manually, the code is doing that automatically.

---

## Models Repository

The <b>SEC-BERT</b> Models are available at HuggingFace and you can load it using the following:

[SEC-BERT-BASE](https://huggingface.co/nlpaueb/sec-bert-base)
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-base")
model = AutoModel.from_pretrained("nlpaueb/sec-bert-base")
```

[SEC-BERT-NUM](https://huggingface.co/nlpaueb/sec-bert-num)
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-num")
model = AutoModel.from_pretrained("nlpaueb/sec-bert-num")
```

[SEC-BERT-BASE](https://huggingface.co/nlpaueb/sec-bert-base)
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-shape")
model = AutoModel.from_pretrained("nlpaueb/sec-bert-shape")
```

Note: You don't need to download or install any model manually, the code is doing that automatically.

---

## Install Python and Project Requirements

It is recommended to create a virtual environment first via Python's venv module or Anaconda's conda.

```commandline
pip install -r requirements.txt
```

```text
click
datasets==2.1.0
gensim==4.2.0
regex
scikit-learn>=1.0.2
seqeval==1.2.2
tensorflow==2.8.0
tensorflow-addons==1.16.1
tf2crf==0.1.24
tokenizers==0.12.1
tqdm
transformers==4.18.0
wandb==0.12.16
wget
```

---

## Running an Experiment
To run an experiment we call the main function `run_experiment.py` located at the root of the project.<br>
We need to provide the following arguments:
* `method`: neural model to run (possible values: `transformer`, `bilstm`)
* `mode`: mode of the experiment. The following modes can be selected:
  * `train`: train a single model
  * `evaluate`: evaluate a pre-trained model 
    
In order to run a train experiment with a `transformer` model we execute:
```commandline
python run_experiment --experiment job_recommendation --method sentence_transformers --mode train
```

___

## Setting up the Experiment's Parameters
We set the parameters of an experiment by editing the configuration file located at the `configurations` folder of the project.<br>
Inside the configurations folder three `json` configuration files (e.g `bilstm.json`, `transformer.json`, `transformer_bilstm.json`) where we can select the parameters of the experiment we would like to run.<br>

If we want to run a `transformer` experiment we need to edit the parameters of `transformer.json`<br>
These parameters are grouped in six groups:
<br><br>
1. `train_parameters`: contains the major parameters of the experiment<br><br>
    * `model_name`: transformer model we would like to train (e.g. `bert-base-uncased`, `sec-bert-base`, `sec-bert-num`, `sec-bert-shape`)
        ``` json
        "model_name": "sec-bert-base"
        ```
      
    * `max_length`: max length in tokens of the input sample.<br><br>
    
    * `replace_numeric_values`: boolean flag indicating wether to replace the numeric values with the special shape token
        ```text
        23.5 -> [XX.X]
        ```
      
    * `subword_pooling`: what subword pooling to perform (possible values are: `all`, `first`, `last`)<br><br>
    
    * `use_fast_tokenizer`: boolean flag indicating wether to use fast tokenizers or not<br><br>
   
2. `general_parameters`: general parameters of the experiment<br><br>
    * `debug`: boolean flag indicating if we want to enable `debug` mode<br>
        During `debug` mode we select only a small portion of the dateset (100 samples for each of the train, validation and test splits), and also enable `tensorflow's eager execution`<br><br>
      
    * `loss_monitor`: loss that the `early stopping` and `reduce learning rate on plateau` `tensorflow's` `callbacks` will monitor<br>
    Possible values are: `val_loss`, `val_micro_f1` and `val_macro_f1`.<br><br>
      
    * `early_stopping_patience`: used by the `early stopping` `tensorflow's` `callback` and indicates the number of epochs to wait without improvement of `loss_monitor` before the training stops.<br><br>
      
    * `reduce_lr_patience`: used by the `reduce learning rate on plateau` `tensorflow's` `callback` and indicates the number of epochs to wait without improvement of loss_monitor before the learning rate is reduced by half<br><br>
      
    * `reduce_lr_cooldown`: used by `reduce learning rate on plateau` `tensorflow's` `callback` and indicates the number of epochs to wait before resuming normal operation after learning rate has been reduced.<br><br>
      
    * `epochs`: maximum number of iterations (epochs) over the corpus. Usually choose a large value and let `early stopping` stop the training after `patience` is reached.<br><br>
      
    * `batch_size`: number of samples per gradient update.<br><br>
   
    * `workers`: workers that create samples during model fit. Choose enough workers to saturate the GPU utilization.<br><br>
      
    * `max_queue_size`: max samples in queue. Choose a large number to saturate the GPU utilization.<br><br>
      
    * `use_multiprocessing`: boolean flag indicating the use of multi-processing for generating samples
    
    * `wandb_entity`: insert your `Weights & Biases` username or team to log the run<br><br>
    
    * `wandb_project`: insert the project's name where the run will be saved.<br><br>
      <br><br>
4. `hyper_parameters`: model hyper-parameters to use when training a single model
    
    * `learning_rate`: learning rate of `Adam` optimizer<br><br>
    
    * `n_layers`: number of stacked `BiLSTM` layers<br><br>
    
    * `n_units`: number of units in each `BiLSTM` layer<br><br>
    
    * `dropout_rate`: randomly sets input units to 0 with a frequency of `dropout_rate`<br><br>
   
    * `crf`: boolean flag indicating the use of CRF layer<br><br>

4. `evaluation`: evaluation parameters of the experiment
   * `pretrained_model`: name of pretrained model used when `evaluate` mode is selected.<br>
   The name is the folder name of the experiment we want to re-evaluate (located at `/data/experiments/runs`) (e.g. `FINER139_2022_01_01_00_00_00`)<br><br>
   ``` json      
      "pretrained_model": "JOB_RECOMMENDATION_2022_03_19_18_42_28"
   ```
   * `splits`: list of dataset splits to evaluate (e.g. `validation`, `test`)<br><br>
---