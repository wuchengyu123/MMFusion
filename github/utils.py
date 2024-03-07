import numpy as np
import sklearn.metrics as metrics
#from imblearn.metrics import sensitivity_score, specificity_score
import pdb
# from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
def compute_metrics(target, output):

    gt_class = target#np.argmax(gt_np, axis=1)
    _,pred_class = output.topk(1, 1, True, True)

    gt_class = gt_class.cpu().detach().numpy()
    pred_class = pred_class.cpu().detach().numpy()


    ACC = accuracy_score(gt_class, pred_class)
    BACC = balanced_accuracy_score(gt_class, pred_class) # balanced accuracy
    Prec = precision_score(gt_class, pred_class, average='macro')
    Rec = recall_score(gt_class, pred_class, average='macro')
    F1 = f1_score(gt_class, pred_class, average='macro')
    #AUC_ovo = metrics.roc_auc_score(gt_class, pred_class.softmax(dim=-1), average='macro', multi_class='ovo')

    #SPEC = specificity_score(gt_class, pred_class, average='macro')

    kappa = cohen_kappa_score(gt_class, pred_class, weights='quadratic')

    # print(confusion_matrix(gt_class, pred_class))
    return ACC, Prec, Rec, F1, kappa


def save_record(ACC, Prec, Rec, F1, kappa, seed, path):
        with open(path,'a+') as f:
            f.write('seed ACC Prec Rec F1 Kappa')
            f.write('\n')
            f.write(str(seed) + ' ')
            f.write(str(ACC) + ' ')
            f.write(str(Prec) + ' ')
            f.write(str(Rec) + ' ')
            f.write(str(F1) + ' ')
            f.write(str(kappa) + ' ')
            f.write('\n')




if __name__ == '__main__':
    import torch
    target = torch.randint(0,4,(12,))
    output = torch.rand(((12,4)),dtype=torch.float32)
    ACC, BACC, Prec, Rec, F1, kappa = compute_metrics(target,output)
    print(ACC)