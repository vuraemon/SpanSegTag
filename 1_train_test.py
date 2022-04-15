#!/usr/bin/env python
# coding: utf-8

# Firstly, we install requirement packages!
# 
# In the case of you have GPU, you will install torch library via `!pip install torch==1.1.0 -f https://download.pytorch.org/whl/cu90/torch_stable.html`

# In[ ]:


get_ipython().system('pip uninstall -y torch torchvision torchtext')
get_ipython().system('pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl')
get_ipython().system('pip install seqeval==0.0.13')
get_ipython().system('pip install tqdm nltk pandas boto3 requests regex psutil')


# Secondly, we train the new model based on some data!

# In[ ]:


get_ipython().system('python wmseg_main.py --do_train                            --train_data_path=../data/sample_data/train.tsv             --eval_data_path=../data/sample_data/dev.tsv                --use_bert                                                  --bert_model=bert-base-chinese                              --n_mlp_span=500                                            --n_mlp_label=500                                           --max_seq_length=300                                        --train_batch_size=16                                       --eval_batch_size=16                                        --num_train_epochs=30                                       --warmup_proportion=0.1                                     --mlp_span_dropout=0.1                                      --learning_rate=1e-5                                        --patient=5                                                 --model_name=sample_model')


# Thirdly, if the training progress is stuck, we restore the training progress!

# In[ ]:


get_ipython().system('python wmseg_main.py --do_train                            --restore_training=1                                        --restore_training_epoch=10                                 --restore_training_saved_model=./models/your_model/model.pt --train_data_path=../data/sample_data/train.tsv             --eval_data_path=../data/sample_data/dev.tsv                --use_bert                                                  --bert_model=bert-base-chinese                              --n_mlp_span=500                                            --n_mlp_label=500                                           --max_seq_length=300                                        --train_batch_size=16                                       --eval_batch_size=16                                        --num_train_epochs=100                                      --warmup_proportion=0.1                                     --mlp_span_dropout=0.1                                      --learning_rate=1e-5                                        --patient=15                                                --model_name=your_new_model')


# We re-evaluate all dev sets. Finally, we evaluate all test sets! The prediction is saved in the our_prediction directory.

# In[ ]:


import subprocess

dataset = 'your_dataset_name'      # For example: CTB5, CTB6, CTB7, CTB9, UD1, UD2, or sample_data
data_set_type = 'test'             # For example: dev/test
your_model_name = 'your_model_name'

output = subprocess.getoutput(f"python wmseg_main.py --do_test --eval_data_path=../data/{dataset}/{data_set_type}.tsv --eval_model=./models/{your_model_name}/model.pt --prediction_path=./our_prediction/{dataset}_BERT_{data_set_type}.txt")
print(output)

