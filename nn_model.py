import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BookImpactPredictor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(BookImpactPredictor, self).__init__()
        self.embedding = nn.EmbeddingBag(input_size, embedding_dim=100)
        self.fc1 = nn.Linear(300, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
        
    def forward(self, title, description, published_year, other_features):
        title_embedding = self.embedding(title)
        description_embedding = self.embedding(description)
        year_embedding = self.embedding(published_year)
        
        x = torch.cat((title_embedding, description_embedding, year_embedding, other_features), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()


