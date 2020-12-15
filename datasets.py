import torch
import numpy as np
from torch.utils.data import Dataset

class DCMQDataSet(Dataset):
    def __init__(self, root_path):
        super().__init__()
        self._path = root_path
        self.data = None
    def Train(self):
        img_path = self._path + 'img_embs_train.npy'
        cap_path = self._path + 'cap_embs_train.npy'
        img_vec = torch.tensor(np.load(img_path))
        cap_vec = torch.tensor(np.load(cap_path))
        self.data = torch.cat([img_vec, cap_vec], dim=0)
        # self.data /= torch.norm(self.data, p=2, dim=1, keepdim=True)
        return self
    def Query(self, mode):
        if mode == 'i2t':
            query_path = self._path + 'img_embs_val.npy'
        else:
            query_path = self._path + 'cap_embs_val.npy'
        query_vec = torch.tensor(np.load(query_path))
        self.data = query_vec
        # self.data /= torch.norm(self.data, p=2, dim=1, keepdim=True)
        return self
    def Retrival(self, mode):
        if mode == 'i2t':
            retrieval_path = self._path + 'cap_embs_val.npy'
        else:
            retrieval_path = self._path + 'img_embs_val.npy'
        retrieval_vec = torch.tensor(np.load(retrieval_path))
        self.data = retrieval_vec
        # self.data /= torch.norm(self.data, p=2, dim=1, keepdim=True)
        return self
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]