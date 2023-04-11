import torch
from tqdm.auto import tqdm
from timeit import default_timer as timer
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

    
def train_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              device: torch.device):
    
   
    model.to(device)
    model.train()
    train_loss = 0
    
    for batch, (X, y) in enumerate(dataloader):
        
        X, y = X.to(device), y.to(device)
        proba = model(X)['clipwise_output']
        loss = loss_fn(proba.squeeze(), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss

    train_loss = train_loss / len(dataloader) 
    
    return train_loss

def test_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device):
     
    preds_list = np.array(0)
    true_labels = np.array(0)
    test_loss = 0
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            proba = model(X)['clipwise_output']
            loss = loss_fn(proba.squeeze(), y)
            test_loss += loss
            preds = proba.round()
            preds_list = np.append(preds_list, preds.cpu().detach().numpy())
            true_labels = np.append(true_labels, y.cpu().detach().numpy())
    
    fscore = f1_score(preds_list[1:], true_labels[1:])
    test_loss = test_loss / len(dataloader)
    
    return test_loss, fscore


def train(model: torch.nn.Module,
         train_dataloader: torch.utils.data.DataLoader,
         test_dataloader:  torch.utils.data.DataLoader,
         loss_fn: torch.nn.Module,
         optimizer: torch.optim.Optimizer,
         epochs: int,
         device: torch.device):
    
    for epoch in tqdm(range(epochs)):
        
        train_loss = train_step(model=model,
                               dataloader=train_dataloader,
                               loss_fn=loss_fn,
                               optimizer=optimizer,
                               device=device)
        
        test_loss, fscore = test_step(model=model,
                                     dataloader=test_dataloader,
                                     loss_fn=loss_fn,
                                     device=device)
        
        if fscore > 0.99:
            print('\n\n')
            print(f"Epoch {epoch}  |  train loss: {train_loss}  |  test loss: {test_loss}  |  f1-score: {fscore}")
            break
        
        print(f"Epoch {epoch}  |  train loss: {train_loss}  |  test loss: {test_loss}  |  f1-score: {fscore}")    
    
    
    
    