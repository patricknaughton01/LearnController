from torch.utils.data import Dataset

class ConcreteDataset(Dataset):
    def __init__(self, data):
        assert len(data) == 3, 'You should provide states, sequence lengths, and number of humans!'
        self.data = data 
    
    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, i):
        return self.data[0][i], self.data[1][i], self.data[2][i]