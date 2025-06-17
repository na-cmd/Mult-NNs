import pandas as pd
import torch
import torch.nn as nn
df = pd.read_csv('output.csv')
col1_values = df.iloc[:, 0].values
col2_values = df.iloc[:, 1].values
device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = torch.tensor([[x, y] for x, y in zip(col1_values, col2_values)], dtype=torch.double, device=device)
Y = torch.tensor(df["label"].values, dtype=torch.double, device=device)
Y = torch.unsqueeze(Y, 0)
Y = Y.transpose(0, 1)
print(X, Y, X.shape, Y.shape)
input = torch.tensor([[210, 100], [912, 445], [9, 24]], dtype=torch.double, device=device)

### y = wx
class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()
    self.input_layer = nn.Linear(2, 200)
    self.hidden_layer = nn.Linear(200, 1)
    self.relu = nn.ReLU()
    self.double()
  def forward(self, pred):
    pred = torch.cat((pred[:,0:1], pred[:,1:2]), dim=1)
    pred = self.input_layer(pred)
    pred = self.relu(pred)
    pred = self.hidden_layer(pred)
    return pred
model = Network().to(device)

#macroparametres
epochs = 400
LR = 0.01
optimizer = torch.optim.Adam(list(model.parameters()), LR )
criterion = nn.MSELoss()

#trainer
for i in range(epochs):
  prediction = torch.abs(model(torch.cat((X, Y), dim=1)))
  loss = criterion(Y, prediction)
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

#save the model
torch.save(model, 'model.pt')