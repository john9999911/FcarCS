# FcarCS
Fine-grained Co-Attentive Representation Learning for Semantic Code Search

`Fine-grained Co-Attentive Representation Learning for Semantic Code Search.pdf` is our paper.

Accepted by SANER 2022: IEEE International Conference on Software Analysis, Evolution, and Reengineering

`FcarCS` is our model proposed in this paper.

`DeepCS` `UNIF` `TabCS` are replication packages of baselines.


## Dependency
> Tested in Ubuntu 16.04
* Python 2.7-3.6
* Keras 2.1.3 or newer
* Tensorflow-gpu 1.7.0


## Usage

   ### DataSets
  The processed datasets used in our paper will be found at https://pan.baidu.com/s/1mrVdCw-iz7ZY-wLIoI-bWg  password:75dl
  
  You can also find the original datasets at https://github.com/xing-hu/EMSE-DeepCom
  
  And the `/data` folder need be included by `/keras`. 
  
  
   ### Get statement-level structure of code
   You can get statement-level structure of code by the source codes in `getStaTree`.
   
   ### Configuration
   
   Edit hyper-parameters and settings in `config.py`
   
   Set reload model/epoch in `config.py`
   
   ### Train and Evaluate
   
   ```bash
   python main.py --mode train
   python main.py --mode eval
