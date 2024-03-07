import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader.Dataset import CustomDataSet
from model.diffusion import ConditionalModel
from model import diffusion_utils
from sklearn.model_selection import train_test_split
from model.BuildModel import BuildModel
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
from utils import *

import time
import numpy as np
print(torch.cuda.is_available())
print(torch.cuda.device_count())
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')


def main(random_seeds,batch_size,X,y,epochs,lr,weight_decay,start_t,end_T,num_classes,num_timesteps):
    acc_arr = []
    pre_arr = []
    f1_arr = []
    rec_arr = []
    for seed in random_seeds:
        print(f'random seed:{seed}')
        print('-'*20)

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=seed)
        train = pd.concat([y_train,X_train],axis=1)
        test = pd.concat([y_test,X_test],axis=1)

        train_data = pd.DataFrame(train.values,columns=train.columns)
        test_data = pd.DataFrame(test.values,columns=test.columns)

        train_set= CustomDataSet(
                csv_data=train_data,
                n_radiomics_col_names=n_rad_col,  # 使用特征列名
                t_radiomics_col_names=t_rad_col,
                blood_col_names=blood_col,
                all_col_names=all_col,
                t_ct_dataset_path= t_image_path,
                n_ct_dataset_path= n_image_path,
                is_test=False
                )

        valid_set = CustomDataSet(
            csv_data=test_data,
            n_radiomics_col_names=n_rad_col,  # 使用特征列名
            t_radiomics_col_names=t_rad_col,
            blood_col_names=blood_col,
            all_col_names=all_col,
            t_ct_dataset_path=t_image_path,
            n_ct_dataset_path=n_image_path,
            is_test=True
        )
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
        valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False)

        guide_model = BuildModel(num_classes).to(device)
        CARD = ConditionalModel(timesteps=num_timesteps,num_classes=num_classes,guidance=True).to(device)

        optimizer = torch.optim.Adam(CARD.parameters(),lr=lr,weight_decay=weight_decay)
        aux_optimizer = torch.optim.Adam(guide_model.parameters(),lr=lr,weight_decay=weight_decay)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_mult=2,T_0=90,eta_min=1e-5) # 8e-6
        aux_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(aux_optimizer,T_mult=2,T_0=80,eta_min=1e-4)

        BCE_criterion = nn.BCELoss()
        MSE_criterion = nn.MSELoss()

        max_acc = 0
        max_pre = 0
        max_rec = 0
        max_f1 = 0
        max_kappa = 0

        for epoch in range(epochs):

            betas = diffusion_utils.make_beta_schedule(schedule="linear", num_timesteps=num_timesteps,
                                                       start=start_t, end=end_T).to(device)
            betas_sqrt = torch.sqrt(betas)
            alphas = 1.0 - betas
            one_minus_betas_sqrt = torch.sqrt(alphas)
            alphas_cumprod = alphas.cumprod(dim=0)
            alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
            one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
            alphas_cumprod_prev = torch.cat(
                [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
            )

            CARD.train()
            guide_model.train()

            total_loss = 0

            guide_pred_train = None
            y_true_train = None

            for batch in train_loader:
                t_img = batch['t_img']
                blood_data = batch['blood_data']
                n_radiomics_data = batch['n_radiomics_data']
                t_radiomics_data = batch['t_radiomics_data']
                n_img_patch0 = batch['n_img_patch0']
                n_img_patch1 = batch['n_img_patch1']
                n_img_patch2 = batch['n_img_patch2']
                target1 = batch['GT_2']

                t_img = t_img.to(device, dtype=torch.float32)
                n_img0 = n_img_patch0.to(device, dtype=torch.float32)
                n_img1 = n_img_patch1.to(device, dtype=torch.float32)
                n_img2 = n_img_patch2.to(device, dtype=torch.float32)
                blood_data = blood_data.to(device, dtype=torch.float32)
                t_radiomics_data = t_radiomics_data.to(device, dtype=torch.float32)
                n_radiomics_data = n_radiomics_data.to(device, dtype=torch.float32)
                target1 = target1.to(device, dtype=torch.int64)

                n = t_img.size(0)
                t = torch.randint(
                    low=0, high=num_timesteps, size=(n // 2 + 1,)
                ).to(device)
                t = torch.cat([t, num_timesteps - 1 - t], dim=0)[:n]

                target = F.one_hot(target1,num_classes=num_classes).float()
                target = target.to(device)

                t_um, t_m, n_um, n_m, feats, yhat = guide_model(t_img=t_img, n_patch0=n_img0, n_patch1=n_img1,
                                                          n_patch2=n_img2,
                                                           blood=blood_data,
                                                          n_radiomics=n_radiomics_data,
                                                          t_radiomics=t_radiomics_data, concat_type='train')


                yhat = yhat.softmax(dim=-1)

                guide_pred_train = torch.cat([guide_pred_train, yhat]) if guide_pred_train is not None else yhat
                y_true_train = torch.cat([y_true_train, target1]) if y_true_train is not None else target1


                aux_cost = BCE_criterion(yhat, target) + MSE_criterion(t_m, t_um) + MSE_criterion(n_m, n_um)

                aux_optimizer.zero_grad()
                aux_cost.backward()
                aux_optimizer.step()
                aux_scheduler.step()

                total_loss += aux_cost.item()

                if epoch > 50:
                    t_um, t_m, n_um, n_m, feats, yhat = guide_model(t_img=t_img, n_patch0=n_img0, n_patch1=n_img1,
                                                                    n_patch2=n_img2,
                                                                    blood=blood_data,
                                                                    n_radiomics=n_radiomics_data,
                                                                    t_radiomics=t_radiomics_data, concat_type='train')
                    e = torch.randn_like(target).to(device)



                    out = diffusion_utils.q_sample(target, yhat.detach(),
                                                   # target: ground truth  yhat: ground truth 的预测   y_t 前向过程最终得到的噪声
                                                   alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise=e)

                    if support == True:
                        out = diffusion_utils.y_0_reparam(CARD,feats.detach(),out,yhat.detach(),yhat.detach(),t,one_minus_alphas_bar_sqrt)


                    out = out.softmax(dim=-1)

                    output = CARD(x=feats.detach(), y=out, t=t, yhat=yhat.detach())

                    guide_pred_train = torch.cat([guide_pred_train, yhat]) if guide_pred_train is not None else yhat
                    y_true_train = torch.cat([y_true_train, target1]) if y_true_train is not None else target1

                    if support == True:
                        loss = 0.1*MSE_criterion(e, output) + 0.9*BCE_criterion(out,target)
                    else:
                        loss = 0.1 * MSE_criterion(e, output)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    #scheduler.step()

                    total_loss += loss.item()

                print('Epoch:{} Guide_loss:{}'.format(epoch, total_loss/len(train_loader)))

                ACC_, Prec_, Rec_, F1_, kappa_ = compute_metrics(y_true_train, guide_pred_train)
                print('Epoch:{} Train: ACC:{:.4f} Prec:{:.4f}, Rec:{:.4f} F1:{:.4f}, Kappa:{:.4f}'.format(epoch, ACC_,
                                                                                                                    Prec_,
                                                                                                                    Rec_, F1_,
                                                                                                                    kappa_))

            CARD.eval()
            guide_model.eval()

            y_true = None
            y_pred = None
            guide_pred = None

            print('start evaluate... epoch:{}'.format(epoch))
            for batch in valid_loader:
                t_img = batch['t_img']
                blood_data = batch['blood_data']
                n_radiomics_data = batch['n_radiomics_data']
                t_radiomics_data = batch['t_radiomics_data']
                n_img_patch0 = batch['n_img_patch0']
                n_img_patch1 = batch['n_img_patch1']
                n_img_patch2 = batch['n_img_patch2']
                target1 = batch['GT_2']
                all_data = batch['all_data']

                t_img = t_img.to(device, dtype=torch.float32)
                n_img0 = n_img_patch0.to(device, dtype=torch.float32)
                n_img1 = n_img_patch1.to(device, dtype=torch.float32)
                n_img2 = n_img_patch2.to(device, dtype=torch.float32)
                blood_data = blood_data.to(device, dtype=torch.float32)
                t_radiomics_data = t_radiomics_data.to(device, dtype=torch.float32)
                n_radiomics_data = n_radiomics_data.to(device, dtype=torch.float32)
                target1 = target1.to(device, dtype=torch.int64)
                all_data = all_data.to(device, dtype=torch.float32)


                with torch.no_grad():
                    _, _, _, _, feats, yhat = guide_model(t_img=t_img, n_patch0=n_img0, n_patch1=n_img1,
                                                              n_patch2=n_img2,
                                                               blood=blood_data,
                                                              n_radiomics=n_radiomics_data,
                                                              t_radiomics=t_radiomics_data, concat_type='train')

                    y_T_mean = yhat.softmax(dim=-1)

                    output = diffusion_utils.p_sample_loop(CARD, feats, yhat, y_T_mean,
                                                           num_timesteps, alphas,
                                                           one_minus_alphas_bar_sqrt,
                                                           only_last_sample=True)

                    output = output.softmax(dim=-1)


                    y_pred = torch.cat([y_pred, output]) if y_pred is not None else output
                    guide_pred = torch.cat([guide_pred, y_T_mean]) if guide_pred is not None else y_T_mean
                    y_true = torch.cat([y_true, target1]) if y_true is not None else target1


            ACC, Prec, Rec, F1,  kappa = compute_metrics(y_true, y_pred)
            ACC1,  Prec1, Rec1, F11, kappa1 = compute_metrics(y_true, guide_pred)


            print('guide-pred: ACC:{:.4f}  Prec:{:.4f}, Rec:{:.4f} F1:{:.4f}, Kappa:{:.4f}'.format( ACC1, Prec1, Rec1, F11, kappa1))

            print('Diff: ACC: {:.4f} Prec:{:.4f}, Rec:{:.4f} F1:{:.4f}, Kappa:{:.4f}'.format(ACC, Prec,Rec,F1,kappa))
            print('========'*10)

            if max_acc < ACC:
                states = [
                    CARD.state_dict(),
                    optimizer.state_dict(),
                    epoch,
                ]
                torch.save(states, os.path.join("./saved_models/seed{}_diff_ckpt_best_acc{:.4f}.pth".format(seed,ACC)))
                max_acc = max(max_acc,ACC)
                max_pre = max(max_pre,Prec)
                max_rec = max(max_rec,Rec)
                max_f1 = max(max_f1, F1)
                max_kappa = max(max_kappa, kappa)


        save_record(max_acc, max_pre, max_rec, max_f1, max_kappa, seed, 'best_record.txt')
        acc_arr.append(max_acc)
        pre_arr.append(max_pre)
        f1_arr.append(max_f1)
        rec_arr.append(max_rec)