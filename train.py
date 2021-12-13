import torch
from dataloader import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchaudio
from model import Model
import pkbar
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

N_ITER = 10000
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')


# https://quokkas.tistory.com/37
class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        # -val_loss 가 클 수록 더 나은 모델, score < best_score ( -val_loss ) 이면, 모델이 더 나아지지 않음. count.
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print(f'Validation loss  ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def l2_loss(output, target):
    return torch.sum((output - target) ** 2) / output.data.nelement()

# Loss 2. Spectral Loss
def SpectralLoss(x, target):
    '''
    output & target : tensor (Amplitude - time)
    outputSpec : Spectrogram of output tensor
    targetSpec : Spectrogram of target tensor
    '''
    n_fft = 400
    outputSpec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, power=None, window_fn=lambda x: torch.hann_window(n_fft, device='cuda'))(x)
    targetSpec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, power=None, window_fn=lambda x: torch.hann_window(n_fft, device='cuda'))(target)
    gap = outputSpec - targetSpec
    n, _, w, h, _ = targetSpec.size()
    # w : frequency scale, h : time scale

    loss = torch.abs(torch.sum(gap.pow(2))/(2*n*w*h))
    return loss

# Loss 3. Frequency Cropped Loss
def FreqCropLoss(x, target) :
    '''
    x, target : spectrogrammed data with librosa.stft & amplitude to db.
    beta & freq array : hyperparameters
    - beta : weight array
        (length : n+1; beta[0]=0 , for convenience of index)
    - freq : partition boundary frequency array
        (length : n+1; freq[0]=0 & freq[n]=sampling rate, same reason with above)

    Default sampling rate of Librosa is 22050Hz -> so i set max Hz value as 11025Hz.
    '''
    beta = [0, 0.3, 0.1, 0.2, 0.1, 0.3]
    freq = np.array([0, 400, 600, 1500, 3000, 11025])/11025
    partitions = len(freq)

    n_fft = 1000
    outputSpec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, power=None, window_fn=lambda x: torch.hann_window(n_fft, device='cuda'))(x)
    targetSpec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, power=None, window_fn=lambda x: torch.hann_window(n_fft, device='cuda'))(target)
    gap = outputSpec - targetSpec
    w, h = targetSpec.size()
    # w : frequency scale, h : time scale

    loss = 0.
    for k in range(1, partitions):
        l = 0.
        start = int(freq[k-1]*w)+1
        if k==1: start=0
        finish = int(freq[k]*w)
        for i in range(start, finish):
            l += torch.sum(gap.pow(2), dim=1)[i]
        loss += l*beta[k]
    loss = torch.abs(loss)/(2*w*h)
    return loss

# torch Document
def train(model, num_iter, num_epochs, checkpoint_dir, loss_func, load_model_path=None, tb_log_path=None, counter=0,
          saved_loss=None, final=False, tag=""):
    load_epoch = 0
    counter = 0
    best_score = None
    min_loss = np.Inf

    checkpoint_dir = checkpoint_dir+tag
    tb_log_path = tb_log_path+tag

    if load_model_path != None:
        print("Load model weight from saved model...")
        model.load_state_dict(torch.load(load_model_path))
        if not final:
            # Epoch, Counter,
            load_epoch = load_model_path[load_model_path.find('ckpt') + 5:load_model_path.rfind("_")]
            load_epoch = int(load_epoch)

            counter = counter
            min_loss = saved_loss
            best_score = -min_loss

        tb = SummaryWriter(tb_log_path)
    else:
        tb = SummaryWriter()

    model = model.to(DEVICE)
    if final:
        print("Final Training With lower LR")
        LR = 0.00001
        BATCH_SIZE = 32
        load_epoch = 0

        counter = 0
        best_score = None
        min_loss = np.Inf
    else:
        LR = 0.0001
        BATCH_SIZE = 16

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    early_stopping = EarlyStopping(patience=20, verbose=True)
    early_stopping.counter = counter
    early_stopping.val_loss_min = min_loss
    early_stopping.best_score = best_score

    print(f"Starting .. {load_epoch}")

    for epoch in range(load_epoch, num_epochs):
        print("")

        # Pure Train
        loss_tot = 0.0

        train_dataset = Dataset(size=num_iter, batch_size=BATCH_SIZE, mode="train")
        dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)
        kbar = pkbar.Kbar(target=len(dataloader), epoch=epoch, num_epochs=num_epochs, width=50, always_stateful=True)

        model.train()
        for i, (mixture, clean) in enumerate(dataloader):
            mixture = mixture.to(DEVICE)
            clean = clean.to(DEVICE)
            optimizer.zero_grad()
            res = model(mixture)
            loss = loss_func(res, clean)
            loss.backward()
            optimizer.step()

            kbar.update(i + 1, values=[("loss", loss)])
            loss_tot += loss.item()
            if (i+1) % 200 == 0:
                checkpoint_prefix = os.path.join(checkpoint_dir, f"ckpt_{epoch + 1}_{i + 1}.pt")
                torch.save(model.state_dict(), checkpoint_prefix)
            tb.add_scalar(f"Training running loss / Epoch :{epoch+1}", loss, i+1)
        tb.add_scalar("Train Loss", loss_tot, epoch)

        # Validation
        valid_loss = []

        eval_dataset = Dataset(mode="val")
        dataloader = DataLoader(eval_dataset, batch_size=1, pin_memory=True, shuffle=True)

        model.eval()
        for i, (mixture, clean) in enumerate(dataloader):
            mixture = mixture.to(DEVICE)
            clean = clean.to(DEVICE)
            res = model(mixture)
            loss = loss_func(res, clean)
            valid_loss.append(loss.cpu().detach().numpy())

        valid_loss = np.mean(np.array(valid_loss))
        tb.add_scalar("Val Loss", valid_loss, epoch)

        checkpoint_prefix = os.path.join(checkpoint_dir, f"ckpt_{epoch + 1}.pt")
        early_stopping.path = checkpoint_prefix
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("")
            print("Early Stopping")
            if final:
                print("")
                print("Training Complete")
            else:
                print("")
                print("final Training")
                train(model, num_iter=num_iter, num_epochs=1000, checkpoint_dir=checkpoint_dir+"/final",
                      loss_func=loss_func, final=True)
            tb.close()
            break

    tb.close()



if __name__ == "__main__":
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        DEVICE = torch.device('cuda')

    checkpoint_dir = './wave_u_net_checkpoints_SPL'

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    torch.manual_seed(0)
    EPOCHS = 1000

    model = Model()

    # Parameters : need to change - For Continue Learning

    MODEL_PATH = "./wave_u_net_checkpoints/ckpt_72.pt"

    TB_PATH = "./runs/train"

    # counter = fin - 71
    # saved_min_loss = 1.41e-7

    # Training Example
    # train(model, 2000, EPOCHS, checkpoint_dir, l2_loss)

    # Continue Training Example
    # Training Example 1
    # train(model, 2000, EPOCHS, checkpoint_dir, l2_loss,
    #       load_model_path=MODEL_PATH, tb_log_path=TB_PATH, counter=counter, saved_loss=saved_min_loss)

    # Final Training Example
    # train(model, 2000, EPOCHS, checkpoint_dir_final, l2_loss,
    #       load_model_path=MODEL_PATH, tb_log_path=TB_2_PATH, final=True)

    # Training
    train(model, 2000, EPOCHS, checkpoint_dir, SpectralLoss, tag="_SPL")

