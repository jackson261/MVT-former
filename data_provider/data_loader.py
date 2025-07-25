import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__read_data__()

    def __read_data__(self):
        
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

       
        data = torch.from_numpy(df_data.values).float().to(self.device)

        if self.scale:
            
            train_data = data[border1s[0]:border2s[0]]
           
           
            self.mean = train_data.mean(dim=0, keepdim=True)
            self.std = train_data.std(dim=0, keepdim=True)
           
            self.std = torch.where(self.std == 0, torch.tensor(1.0, device=self.device), self.std)
            data = (data - self.mean) / self.std
        else:
           
            self.mean = torch.zeros(data.shape[1], device=self.device)
            self.std = torch.ones(data.shape[1], device=self.device)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            
            months = torch.tensor(df_stamp.date.apply(lambda row: row.month, 1).values, dtype=torch.float32, device=self.device)
            days = torch.tensor(df_stamp.date.apply(lambda row: row.day, 1).values, dtype=torch.float32, device=self.device)
            weekdays = torch.tensor(df_stamp.date.apply(lambda row: row.weekday(), 1).values, dtype=torch.float32, device=self.device)
            hours = torch.tensor(df_stamp.date.apply(lambda row: row.hour, 1).values, dtype=torch.float32, device=self.device)
            data_stamp = torch.stack([months, days, weekdays, hours], dim=1)
        elif self.timeenc == 1:
           
            data_stamp_np = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = torch.from_numpy(data_stamp_np).float().to(self.device).transpose(0, 1)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
       
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")
       
        data = data.to(self.mean.device)
        if self.scale:
            return data * self.std + self.mean
        return data


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
       
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__read_data__()

    def __read_data__(self):
        
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

       
        data = torch.from_numpy(df_data.values).float().to(self.device)

        if self.scale:
            
            train_data = data[border1s[0]:border2s[0]]
            
            self.mean = train_data.mean(dim=0, keepdim=True)
            self.std = train_data.std(dim=0, keepdim=True)
           
            self.std = torch.where(self.std == 0, torch.tensor(1.0, device=self.device), self.std)
            data = (data - self.mean) / self.std
        else:
            
            self.mean = torch.zeros(data.shape[1], device=self.device)
            self.std = torch.ones(data.shape[1], device=self.device)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            
            days = torch.tensor(df_stamp.date.apply(lambda row: row.day, 1).values, dtype=torch.float32, device=self.device)
            weekdays = torch.tensor(df_stamp.date.apply(lambda row: row.weekday(), 1).values, dtype=torch.float32, device=self.device)
            hours = torch.tensor(df_stamp.date.apply(lambda row: row.hour, 1).values, dtype=torch.float32, device=self.device)
            minutes = torch.tensor(df_stamp.date.apply(lambda row: row.minute, 1).map(lambda x: x // 15).values, dtype=torch.float32, device=self.device)
            data_stamp = torch.stack([months, days, weekdays, hours, minutes], dim=1)
        elif self.timeenc == 1:
            
            data_stamp_np = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = torch.from_numpy(data_stamp_np).float().to(self.device).transpose(0, 1)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")
        
        data = data.to(self.mean.device)
        if self.scale:
            return data * self.std + self.mean
        return data


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.__read_data__()

    def __read_data__(self):
        
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        
        if 'date' not in df_raw.columns or self.target not in df_raw.columns:
            raise ValueError("CSV file must contain 'date' and target columns.")

       
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        # 数据集划分
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

       
        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        data = torch.from_numpy(df_data.values).float().to(self.device)


       
        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.mean = train_data.mean(dim=0, keepdim=True) 
            self.std = train_data.std(dim=0, keepdim=True)    
           
            self.std = torch.where(self.std == 0, torch.tensor(1.0, device=self.device), self.std)
            data = (data - self.mean) / self.std
        else:
           
            self.mean = torch.zeros(data.shape[1], device=self.device)
            self.std = torch.ones(data.shape[1], device=self.device)

        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

       
        dates = pd.to_datetime(df_raw['date'][border1:border2])  
        if self.timeenc == 0:
            months = torch.tensor(dates.dt.month.values, dtype=torch.float32, device=self.device)
            days = torch.tensor(dates.dt.day.values, dtype=torch.float32, device=self.device)
            weekdays = torch.tensor(dates.dt.weekday.values, dtype=torch.float32, device=self.device)
            hours = torch.tensor(dates.dt.hour.values, dtype=torch.float32, device=self.device)
            data_stamp = torch.stack([months, days, weekdays, hours], dim=1)
        elif self.timeenc == 1:          
            dates_index = pd.DatetimeIndex(dates)
            data_stamp_np = time_features(dates_index, freq=self.freq)
            data_stamp = torch.from_numpy(data_stamp_np).float().to(self.device).transpose(0, 1)

        self.data_stamp = data_stamp
        



    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
    
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")
       
        data = data.to(self.mean.device)
        if self.scale:
            return data * self.std + self.mean
        return data



class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__read_data__()

    def __read_data__(self):
        
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        
        data = torch.from_numpy(data).float().to(self.device)
        if self.scale:
        
            train_data = torch.from_numpy(train_data).float().to(self.device)
            
            self.mean = train_data.mean(dim=0, keepdim=True)
            self.std = train_data.std(dim=0, keepdim=True)
            self.std = torch.where(self.std == 0, torch.tensor(1.0, device=self.device), self.std)
            data = (data - self.mean) / self.std
        else:
            self.mean = torch.zeros(data.shape[1], device=self.device)
            self.std = torch.ones(data.shape[1], device=self.device)

        
        df = pd.DataFrame(data.cpu().numpy())
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        data = torch.from_numpy(df).float().to(self.device)

        self.data_x = data
        self.data_y = data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1), device=self.device)
        seq_y_mark = torch.zeros((seq_x.shape[0], 1), device=self.device)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")
        data = data.to(self.mean.device)
        if self.scale:
            return data * self.std + self.mean
        return data


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__read_data__()


    def __read_data__(self):
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values
        data = torch.from_numpy(df_data).float().to(self.device)

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.mean = train_data.mean(dim=0, keepdim=True)
            self.std = train_data.std(dim=0, keepdim=True)
            self.std = torch.where(self.std == 0, torch.tensor(1.0, device=self.device), self.std)
            data = (data - self.mean) / self.std
        else:
            self.mean = torch.zeros(data.shape[1], device=self.device)
            self.std = torch.ones(data.shape[1], device=self.device)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1), device=self.device)
        seq_y_mark = torch.zeros((seq_x.shape[0], 1), device=self.device)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")
        data = data.to(self.mean.device)
        if self.scale:
            return data * self.std + self.mean
        return data


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        data = torch.from_numpy(df_data.values).float().to(self.device)

        if self.scale:
            self.mean = data.mean(dim=0, keepdim=True)
            self.std = data.std(dim=0, keepdim=True)
            self.std = torch.where(self.std == 0, torch.tensor(1.0, device=self.device), self.std)
            data = (data - self.mean) / self.std
        else:
            self.mean = torch.zeros(data.shape[1], device=self.device)
            self.std = torch.ones(data.shape[1], device=self.device)

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            months = torch.tensor(df_stamp.date.apply(lambda row: row.month, 1).values, dtype=torch.float32, device=self.device)
            days = torch.tensor(df_stamp.date.apply(lambda row: row.day, 1).values, dtype=torch.float32, device=self.device)
            weekdays = torch.tensor(df_stamp.date.apply(lambda row: row.weekday(), 1).values, dtype=torch.float32, device=self.device)
            hours = torch.tensor(df_stamp.date.apply(lambda row: row.hour, 1).values, dtype=torch.float32, device=self.device)
            minutes = torch.tensor(df_stamp.date.apply(lambda row: row.minute, 1).map(lambda x: x // 15).values, dtype=torch.float32, device=self.device)
            data_stamp = torch.stack([months, days, weekdays, hours, minutes], dim=1)
        elif self.timeenc == 1:
            data_stamp_np = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = torch.from_numpy(data_stamp_np).float().to(self.device).transpose(0, 1)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = torch.from_numpy(df_data.values[border1:border2]).float().to(self.device)
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")
        data = data.to(self.mean.device)
        if self.scale:
            return data * self.std + self.mean
        return data
