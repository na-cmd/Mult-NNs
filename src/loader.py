MAX_VAL_EXP = 3
model = torch.load("model.pt", weights_only=False)
model.eval()
with torch.no_grad():
  pred = model(input/10**MAX_VAL_EXP)
  print(input, torch.abs(pred)*10**(MAX_VAL_EXP*2))