
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
│    ├── p225_001.wav  
│    ├── p225_002.wav  
│    └── ...  
└──dataset_list.txt
</pre>
## 2. Train

run train.py

Modify path : log_tb_path, checkpoint_dir is path of tensorboard log data, checkpointdata

<pre>
Example
model = Model()
train(model, 2000, EPOCHS, checkpoint_dir, SpectralLoss, tag="_SPL")
</pre>

* If you want continue learning
  * Modify suitable load_model_path
  * Modify counter, saved_min_loss for continue early stopping
<pre>
  Example
  # Continue Training Example
  model = Model()

  MODEL_PATH = "./wave_u_net_checkpoints_SPL/ckpt_25.pt"
  TB_PATH = "./runs/train"

  counter = fin - 71
  saved_min_loss = 1.41e-7

  train(model, 2000, EPOCHS, checkpoint_dir, l2_loss,
        load_model_path=MODEL_PATH, tb_log_path=TB_PATH, counter=counter, saved_loss=saved_min_loss, tag="")
</pre>


* If you want separate final learning
  * Modify checkpoint_dir , log_tb_path
  * final=True
<pre>
  Example
  # Final Training Example
  MODEL_PATH = "./wave_u_net_checkpoints_SPL/ckpt_25.pt"

  train(model, 2000, EPOCHS, checkpoint_dir, SpectralLoss,
        load_model_path=MODEL_PATH, final=True)

</pre>

## 3. Evaluation
In this project, Evaluation results saved as tensorboard log data  
for Default, random (10) sampling from clean/noisy testset data

run metrics_tensorboard.py

Modify path : DEFAULT_TB_LOG, models  
   * DEFAULT_TB_LOG : directory for save tensorboard log  
   * model : directory for save tensorboard log

Default Directory : Needed testset's sorted txt file. ( run metrics.py you can get )
<pre>
testset    
├── clean  
│    ├── p232_001.wav  
│    ├── p232_002.wav  
│    └── ...   
└── noisy   
│    ├── p232_001.wav  
│    ├── p232_002.wav  
│    └── ...  
└──testset_list.txt
└──testset_sort_pesq.txt
└──testset_sort_ssnr.txt
</pre>

Default setting

<pre>
    DEFAULT_TB_LOG = "./runs/metrics"
    models = get_ckpts("./wave_u_net_checkpoints/")  
    sampling_num = 10
</pre>

## Extra : Run tensorboard

### 1. Run anaconda prompt, activate project environment
### 2. Execute below command
<pre>
 tensorboard --logdir="Your Tensorboard Path"
</pre>

