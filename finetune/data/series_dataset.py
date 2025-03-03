import os
from data.base_dataset import BasicClassificationDataset


class SeriesClassificationDataset(BasicClassificationDataset):

    def __init__(self, root_dir, transform=None):
        super(SeriesClassificationDataset, self).__init__(root_dir, transform)
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
            
    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        data_dict.update({'image_name': self.image_name_dict['image_name'][index], 'study_series_name': self.image_name_dict['study_series_name'][index]})
        return data_dict
