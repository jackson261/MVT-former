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
        # 定义设备为 GPU 或 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__read_data__()

    def __read_data__(self):
        #self.scaler = StandardScaler()
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

        # 转换为 PyTorch 张量并移动到 GPU
        data = torch.from_numpy(df_data.values).float().to(self.device)

        if self.scale:
            #train_data = df_data[border1s[0]:border2s[0]]
            train_data = data[border1s[0]:border2s[0]]
            # 在 GPU 上计算均值和标准差
            #self.scaler.fit(train_data.values)
            #data = self.scaler.transform(df_data.values)
            self.mean = train_data.mean(dim=0, keepdim=True)
            self.std = train_data.std(dim=0, keepdim=True)
            # 避免除以零
            self.std = torch.where(self.std == 0, torch.tensor(1.0, device=self.device), self.std)
            data = (data - self.mean) / self.std
        else:
            #data = df_data.values
            # 如果不标准化，初始化 mean 和 std
            self.mean = torch.zeros(data.shape[1], device=self.device)
            self.std = torch.ones(data.shape[1], device=self.device)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            #df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            #df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            #df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            #df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            #data_stamp = df_stamp.drop(['date'], 1).values
            months = torch.tensor(df_stamp.date.apply(lambda row: row.month, 1).values, dtype=torch.float32, device=self.device)
            days = torch.tensor(df_stamp.date.apply(lambda row: row.day, 1).values, dtype=torch.float32, device=self.device)
            weekdays = torch.tensor(df_stamp.date.apply(lambda row: row.weekday(), 1).values, dtype=torch.float32, device=self.device)
            hours = torch.tensor(df_stamp.date.apply(lambda row: row.hour, 1).values, dtype=torch.float32, device=self.device)
            data_stamp = torch.stack([months, days, weekdays, hours], dim=1)
        elif self.timeenc == 1:
            #data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            #data_stamp = data_stamp.transpose(1, 0)
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
        #return self.scaler.inverse_transform(data)
        # 确保输入是张量
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")
        # 确保数据与 mean/std 在同一设备上
        data = data.to(self.mean.device)
        if self.scale:
            return data * self.std + self.mean
        return data


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
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
        # 定义设备为 GPU 或 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__read_data__()

    def __read_data__(self):
        #self.scaler = StandardScaler()
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

        # 转换为 PyTorch 张量并移动到 GPU
        data = torch.from_numpy(df_data.values).float().to(self.device)

        if self.scale:
            #train_data = df_data[border1s[0]:border2s[0]]
            #self.scaler.fit(train_data.values)
            #data = self.scaler.transform(df_data.values)
            train_data = data[border1s[0]:border2s[0]]
            # 在 GPU 上计算均值和标准差
            self.mean = train_data.mean(dim=0, keepdim=True)
            self.std = train_data.std(dim=0, keepdim=True)
            # 避免除以零
            self.std = torch.where(self.std == 0, torch.tensor(1.0, device=self.device), self.std)
            data = (data - self.mean) / self.std
        else:
            #data = df_data.values
            # 如果不标准化，初始化 mean 和 std
            self.mean = torch.zeros(data.shape[1], device=self.device)
            self.std = torch.ones(data.shape[1], device=self.device)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            #df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            #df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            #df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            #df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            #df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            #df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            #data_stamp = df_stamp.drop(['date'], 1).values
            days = torch.tensor(df_stamp.date.apply(lambda row: row.day, 1).values, dtype=torch.float32, device=self.device)
            weekdays = torch.tensor(df_stamp.date.apply(lambda row: row.weekday(), 1).values, dtype=torch.float32, device=self.device)
            hours = torch.tensor(df_stamp.date.apply(lambda row: row.hour, 1).values, dtype=torch.float32, device=self.device)
            minutes = torch.tensor(df_stamp.date.apply(lambda row: row.minute, 1).map(lambda x: x // 15).values, dtype=torch.float32, device=self.device)
            data_stamp = torch.stack([months, days, weekdays, hours, minutes], dim=1)
        elif self.timeenc == 1:
            #data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            #data_stamp = data_stamp.transpose(1, 0)
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
        #return self.scaler.inverse_transform(data)
        # 确保输入是张量
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")
        # 确保数据与 mean/std 在同一设备上
        data = data.to(self.mean.device)
        if self.scale:
            return data * self.std + self.mean
        return data


class Dataset_Custom(Dataset):
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

        # 定义设备为 GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 调用数据读取函数
        self.__read_data__()

    def __read_data__(self):
        # 读取数据
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 检查列名是否存在
        if 'date' not in df_raw.columns or self.target not in df_raw.columns:
            raise ValueError("CSV file must contain 'date' and target columns.")

        # 调整列顺序：['date', ...(other features), target]
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

        # 选择特征列并转换为 Torch Tensor
        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        #data = torch.from_numpy(df_data.values).float()
        data = torch.from_numpy(df_data.values).float().to(self.device)


        # 手动标准化
        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.mean = train_data.mean(dim=0, keepdim=True)  # 计算训练数据的均值
            self.std = train_data.std(dim=0, keepdim=True)    # 计算训练数据的标准差
            # 避免除以零
            #self.std = torch.where(self.std == 0, torch.tensor(1.0), self.std)
            self.std = torch.where(self.std == 0, torch.tensor(1.0, device=self.device), self.std)
            data = (data - self.mean) / self.std
        else:
            # 如果不标准化，初始化 mean 和 std 以便逆标准化
            #self.mean = torch.zeros(data.shape[1])
            #self.std = torch.ones(data.shape[1])
            self.mean = torch.zeros(data.shape[1], device=self.device)
            self.std = torch.ones(data.shape[1], device=self.device)

        # 存储为 CPU Tensor
        #self.data_x = data[border1:border2].to('cpu')
        #self.data_y = data[border1:border2].to('cpu')
        #self.mean = self.mean.to('cpu')
        #self.std = self.std.to('cpu')
        # 存储为 GPU Tensor
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # 时间特征处理（优化以减少 pandas 依赖）
        dates = pd.to_datetime(df_raw['date'][border1:border2])  # 保持为 pd.Series
        if self.timeenc == 0:
            # 手动提取时间特征
            #months = np.array([date.month for date in dates])
            #days = np.array([date.day for date in dates])
            #weekdays = np.array([date.weekday() for date in dates])
            #hours = np.array([date.hour for date in dates])
            #data_stamp = np.stack([months, days, weekdays, hours], axis=1)
            
            #months = dates.dt.month.values
            #days = dates.dt.day.values
            #weekdays = dates.dt.weekday.values
            #hours = dates.dt.hour.values
            #data_stamp = np.stack([months, days, weekdays, hours], axis=1)
            months = torch.tensor(dates.dt.month.values, dtype=torch.float32, device=self.device)
            days = torch.tensor(dates.dt.day.values, dtype=torch.float32, device=self.device)
            weekdays = torch.tensor(dates.dt.weekday.values, dtype=torch.float32, device=self.device)
            hours = torch.tensor(dates.dt.hour.values, dtype=torch.float32, device=self.device)
            data_stamp = torch.stack([months, days, weekdays, hours], dim=1)
        elif self.timeenc == 1:
            #data_stamp = time_features(dates, freq=self.freq)
            #data_stamp = data_stamp.transpose(1, 0)
            # 转换为 pd.DatetimeIndex
            dates_index = pd.DatetimeIndex(dates)
            #data_stamp = time_features(dates_index, freq=self.freq)
            data_stamp_np = time_features(dates_index, freq=self.freq)
            # 转换为 PyTorch 张量并移动到 GPU
            #data_stamp = data_stamp.transpose(1, 0)
            # # 不再需要 transpose，因为 time_features 已返回正确的形状
            data_stamp = torch.from_numpy(data_stamp_np).float().to(self.device).transpose(0, 1)

        # 转换为 Torch Tensor 并移动到设备
        #self.data_stamp = torch.from_numpy(data_stamp).float().to(device)
        # 修复：将 self.data_stamp 存储为 CPU Tensor
        #self.data_stamp = torch.from_numpy(data_stamp).float().to('cpu')
        # 时间特征存储为 GPU Tensor
        #self.data_stamp = torch.from_numpy(data_stamp).float().to(self.device)
        self.data_stamp = data_stamp
        # 将时间特征转换为 Torch Tensor 并移动到 GPU
        #self.data_stamp = torch.tensor(data_stamp, dtype=torch.float32, device=self.device)



    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # 确保数据在 GPU 上
        #seq_x = self.data_x[s_begin:s_end].to(self.device)
        #seq_y = self.data_y[r_begin:r_end].to(self.device)
        #seq_x_mark = self.data_stamp[s_begin:s_end].to(self.device)
        #seq_y_mark = self.data_stamp[r_begin:r_end].to(self.device)


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        # 逆标准化
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")
        # 确保数据与 mean/std 在同一设备上
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
        # 定义设备为 GPU 或 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__read_data__()

    def __read_data__(self):
        #self.scaler = StandardScaler()
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

        # 转换为 PyTorch 张量并移动到 GPU
        data = torch.from_numpy(data).float().to(self.device)
        if self.scale:
            #self.scaler.fit(train_data)
            #data = self.scaler.transform(data)
            train_data = torch.from_numpy(train_data).float().to(self.device)
            # 在 GPU 上计算均值和标准差
            self.mean = train_data.mean(dim=0, keepdim=True)
            self.std = train_data.std(dim=0, keepdim=True)
            # 避免除以零
            self.std = torch.where(self.std == 0, torch.tensor(1.0, device=self.device), self.std)
            data = (data - self.mean) / self.std
        else:
            # 如果不标准化，初始化 mean 和 std
            self.mean = torch.zeros(data.shape[1], device=self.device)
            self.std = torch.ones(data.shape[1], device=self.device)

        #df = pd.DataFrame(data)
        #df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        # 由于 fillna 需要 Pandas，先将数据移到 CPU 处理
        df = pd.DataFrame(data.cpu().numpy())
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        # 再将处理后的数据移回 GPU
        data = torch.from_numpy(df).float().to(self.device)

        #self.data_x = df
        #self.data_y = df
        self.data_x = data
        self.data_y = data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        #seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        #seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        seq_x_mark = torch.zeros((seq_x.shape[0], 1), device=self.device)
        seq_y_mark = torch.zeros((seq_x.shape[0], 1), device=self.device)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        #return self.scaler.inverse_transform(data)
        # 确保输入是张量
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")
        # 确保数据与 mean/std 在同一设备上
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
        # 定义设备为 GPU 或 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__read_data__()


    def __read_data__(self):
        #self.scaler = StandardScaler()
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
        # 转换为 PyTorch 张量并移动到 GPU
        data = torch.from_numpy(df_data).float().to(self.device)

        if self.scale:
            #train_data = df_data[border1s[0]:border2s[0]]
            #self.scaler.fit(train_data)
            #data = self.scaler.transform(df_data)
            train_data = data[border1s[0]:border2s[0]]
            # 在 GPU 上计算均值和标准差
            self.mean = train_data.mean(dim=0, keepdim=True)
            self.std = train_data.std(dim=0, keepdim=True)
            # 避免除以零
            self.std = torch.where(self.std == 0, torch.tensor(1.0, device=self.device), self.std)
            data = (data - self.mean) / self.std
        else:
            #data = df_data
            # 如果不标准化，初始化 mean 和 std
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
        #seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        #seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        seq_x_mark = torch.zeros((seq_x.shape[0], 1), device=self.device)
        seq_y_mark = torch.zeros((seq_x.shape[0], 1), device=self.device)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        #return self.scaler.inverse_transform(data)
        # 确保输入是张量
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")
        # 确保数据与 mean/std 在同一设备上
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
        # 定义设备为 GPU 或 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__read_data__()

    def __read_data__(self):
        #self.scaler = StandardScaler()
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
        # 转换为 PyTorch 张量并移动到 GPU
        data = torch.from_numpy(df_data.values).float().to(self.device)

        if self.scale:
            #self.scaler.fit(df_data.values)
            #data = self.scaler.transform(df_data.values)
            # 在 GPU 上计算均值和标准差
            self.mean = data.mean(dim=0, keepdim=True)
            self.std = data.std(dim=0, keepdim=True)
            # 避免除以零
            self.std = torch.where(self.std == 0, torch.tensor(1.0, device=self.device), self.std)
            data = (data - self.mean) / self.std
        else:
            #data = df_data.values
            # 如果不标准化，初始化 mean 和 std
            self.mean = torch.zeros(data.shape[1], device=self.device)
            self.std = torch.ones(data.shape[1], device=self.device)

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            #df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            #df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            #df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            #df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            #df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            #df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            #data_stamp = df_stamp.drop(['date'], 1).values
            months = torch.tensor(df_stamp.date.apply(lambda row: row.month, 1).values, dtype=torch.float32, device=self.device)
            days = torch.tensor(df_stamp.date.apply(lambda row: row.day, 1).values, dtype=torch.float32, device=self.device)
            weekdays = torch.tensor(df_stamp.date.apply(lambda row: row.weekday(), 1).values, dtype=torch.float32, device=self.device)
            hours = torch.tensor(df_stamp.date.apply(lambda row: row.hour, 1).values, dtype=torch.float32, device=self.device)
            minutes = torch.tensor(df_stamp.date.apply(lambda row: row.minute, 1).map(lambda x: x // 15).values, dtype=torch.float32, device=self.device)
            data_stamp = torch.stack([months, days, weekdays, hours, minutes], dim=1)
        elif self.timeenc == 1:
            #data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            #data_stamp = data_stamp.transpose(1, 0)
            data_stamp_np = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = torch.from_numpy(data_stamp_np).float().to(self.device).transpose(0, 1)

        self.data_x = data[border1:border2]
        if self.inverse:
            #self.data_y = df_data.values[border1:border2]
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
        #return self.scaler.inverse_transform(data)
        # 确保输入是张量
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")
        # 确保数据与 mean/std 在同一设备上
        data = data.to(self.mean.device)
        if self.scale:
            return data * self.std + self.mean
        return data
