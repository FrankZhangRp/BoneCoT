import os
import pandas as pd
import numpy as np
import torch
from PIL import Image

class O1ClassificationDataset(object):
    def __init__(self, data_path=None, transform=None, logger=None):
        assert os.path.exists(data_path), '{} not exist'.format(data_path)
        self.data_path = data_path
        self.transform = transform
        self.flie_type = data_path.split('.')[-1]
        self.logger = logger
        
        self.images_list, self.labels_list, self.features_list, self.ori_csv = self._read_data()
        
        self.feature_dim = len(self.features_list[0])
        self.image_name_dict = {
            'image_name': [],
            'study_series_name': []
        }
        self.series_name_list = []
        self.get_image_list()
    
    def get_image_list(self):
        series_name_list = set()         
        for image_path, label in zip(self.images_list, self.labels_list):
            image_name = os.path.basename(image_path)
            self.image_name_dict['image_name'].append(image_name)
            
            split_image_name = image_name.split('_')
            study_id = split_image_name[0]
            series_name = split_image_name[1]

            study_series_name = study_id + '_' + series_name
            self.image_name_dict['study_series_name'].append(study_series_name)
            series_name_list.add(study_series_name)

        self.series_name_list = list(series_name_list)
        
    def _read_data(self):
        if self.flie_type == 'csv':
            file_df = pd.read_csv(self.data_path)
            images_list = file_df['img_path'].values
            labels_list = file_df['label'].values
            features_list = []
            for i in range(len(file_df)):
                feature = file_df.iloc[i, 2:].values
                feature = np.array(feature, dtype=np.float32)
                features_list.append(feature)
        else:
            raise ValueError('O1 only accept csv file')
        return images_list, labels_list, features_list, file_df
    
    def __getitem__(self, index):
        image_path = self.images_list[index]
        label = self.labels_list[index]
        feature = self.features_list[index]
        feature = torch.tensor(feature, dtype=torch.float32)
        image = Image.open(image_path)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        data_dict = {'image': image, 'label': label, 'image_name': self.image_name_dict['image_name'][index], 'feature': feature, 'study_series_name': self.image_name_dict['study_series_name'][index]}
        return data_dict
    
    def __len__(self):
        return len(self.images_list)


