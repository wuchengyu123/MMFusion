import os.path
import pandas as pd
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
import torchio as tio

class CustomDataSet(Dataset):
    def __init__(self,
                 csv_data,
                 n_radiomics_col_names,
                 t_radiomics_col_names,
                 blood_col_names,
                 all_col_names,
                 t_ct_dataset_path,
                 n_ct_dataset_path,
                 is_test
                 ):
        super(CustomDataSet, self).__init__()
        self.transform = tio.Compose([
            tio.RandomNoise(p=0.4),
            tio.RandomBiasField(p=0.3),
            tio.RandomFlip(p=0.6),
            tio.RandomMotion(p=0.2),
        ])
        self.t_ct_dataset_path = t_ct_dataset_path
        self.n_ct_dataset_path = n_ct_dataset_path
        self.n_radiomics_col_names = n_radiomics_col_names
        self.t_radiomics_col_names = t_radiomics_col_names
        self.blood_col_names = blood_col_names
        self.clinical_col_names = all_col_names
        self.csv_data = csv_data

        self.all_data = self.csv_data[[*all_col_names]].values
        self.blood_data = self.csv_data[[*blood_col_names]].values
        self.n_radiomics_data = self.csv_data[[*n_radiomics_col_names]].values
        self.t_radiomics_data = self.csv_data[[*t_radiomics_col_names]].values

        self.is_test = is_test

        self.csv_IBEX_CT_NAME = np.squeeze(self.csv_data.loc[:,['IBEX_CT_NAME']].values)


    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index):
        all_data = np.asarray(self.all_data[index],dtype=np.float32)
        blood_data = np.asarray(self.blood_data[index], dtype=np.float32)
        n_radiomics_data = np.asarray(self.n_radiomics_data[index], dtype=np.float32)
        t_radiomics_data = np.asarray(self.t_radiomics_data[index], dtype=np.float32)

        GT = np.asarray(self.csv_data.loc[index,'GT_2'],dtype=np.float32)
        PatientID = np.asarray(self.csv_data.loc[index, 'PatientID'], dtype=np.int32)


        t_img = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(self.t_ct_dataset_path,
                               '{}_CT.nii.gz'.format(str(self.csv_IBEX_CT_NAME[index])))))

        n_patch0 = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(self.n_ct_dataset_path,
                               '{}_patch0.nii.gz'.format(str(self.csv_IBEX_CT_NAME[index])))))
        n_patch1 = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(self.n_ct_dataset_path,
                                        '{}_patch1.nii.gz'.format(str(self.csv_IBEX_CT_NAME[index])))))
        n_patch2 = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(self.n_ct_dataset_path,
                                        '{}_patch2.nii.gz'.format(str(self.csv_IBEX_CT_NAME[index])))))

        t_img = np.expand_dims(t_img,axis=0)
        n_patch0 = np.expand_dims(n_patch0,axis=0)
        n_patch1 = np.expand_dims(n_patch1,axis=0)
        n_patch2 = np.expand_dims(n_patch2,axis=0)

        t_img = np.asarray(t_img,dtype=np.float32)
        n_patch0 = np.asarray(n_patch0,dtype=np.float32)
        n_patch1 = np.asarray(n_patch1,dtype=np.float32)
        n_patch2 = np.asarray(n_patch2,dtype=np.float32)

        if not self.is_test:
            t_img = self.transform(t_img)
            n_patch0 = self.transform(n_patch0)
            n_patch1 = self.transform(n_patch1)
            n_patch2 = self.transform(n_patch2)

        return  {'t_img':t_img,'n_img_patch0':n_patch0,'n_img_patch1':n_patch1,'n_img_patch2':n_patch2,
                 'GT_2': GT, 'n_radiomics_data':n_radiomics_data, 'all_data':all_data,'PatientID':PatientID,
                 't_radiomics_data':t_radiomics_data,'blood_data':blood_data}
