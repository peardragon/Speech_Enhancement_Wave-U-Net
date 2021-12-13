# Speech_Enhancement_Wave-U-Net
  
References  
  
Model : https://github.com/noise-suppression/Wave-U-Net-for-Speech-Enhancement/blob/master/model/unet_basic.py  
Eval in Metrics : https://github.com/jdavibedoya/SE_Wave-U-Net/blob/d54aa19245cbe40de10ab2c90486043aad265bb3/Metrics.py  
Early Stopping : https://quokkas.tistory.com/37  

# How to Use

## 1. Dataset

txt file make for Custom Dataloader

1st cell of file editor.ipynb or dataloader.py run in main

* Dataset Dir  
<pre>
dataset    
├── clean  
│    ├── p225_001.wav  
│    ├── p225_002.wav  
│    └── ...   
└── noisy   
     ├── p225_001.wav  
     ├── p225_002.wav  
     └── ...  
</pre>
## 2. Train

run train.py

Modify path : log_tb_path, checkpoint_dir is path of tensorboard log data, checkpointdata

* If you want continue learning
  * Modify suitable load_model_path
  * Modify counter, saved_min_loss for continue early stopping
<pre>
  Example
  # Continue Training Example

  counter = fin - 71
  saved_min_loss = 1.41e-7

  train(model, 2000, EPOCHS, checkpoint_dir, l2_loss,
        load_model_path=MODEL_PATH, tb_log_path=TB_PATH, counter=counter, saved_loss=saved_min_loss)
</pre>


* If you want separate final learning
  * Modify checkpoint_dir , log_tb_path
  * final=True
<pre>
  Example
  # Final Training Example

  train(model, 2000, EPOCHS, checkpoint_dir_final, l2_loss,
  load_model_path=MODEL_PATH, tb_log_path=TB_2_PATH, final=True)
</pre>

## 3. Evaluation
In this project, Evaluation results saved as tensorboard log data

run metrics_tensorboard.py

Modify path : DEFAULT_TB_LOG, sample_audio is path of tensorboard path for save evaluation results, 
sample_audio name  

Default setting

<pre>
    DEFAULT_TB_LOG = "./runs/metrics"
    sample_audio = "p232_023.wav"
</pre>