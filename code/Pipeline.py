import torch

def Train(device, model, dataloader, criterion, optimizer):
  model.train()

  running_loss        = 0.0
  
  for i, x in enumerate(dataloader):
      x = x.to(device)

      optimizer.zero_grad()

      _, reconstructed = model(x)

      loss        =  criterion(reconstructed, x) 
      
      running_loss        += loss.item()
      
      loss.backward()
      optimizer.step()

  del x
  torch.cuda.empty_cache()

  running_loss /= len(dataloader)

  return running_loss

def Valid(device, model, dataloader, n_components=2):
  model.eval()

  result = torch.zeros(len(dataloader), n_components, device=device)
  for i, x in enumerate(dataloader):
      x = x.to(device)
      embedded, _ = model(x)
      result[i, :] = embedded

  return result