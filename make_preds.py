import torch
import numpy as np
from data_setup import SandDataTest
from torch.utils.data import DataLoader
import pandas as pd
import model_builder
import warnings

warnings.simplefilter('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
df_test = pd.read_csv('data/test.csv')


def make_preds(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               device: torch.device):
    preds_list = np.array(0)
    # put model into eval mode
    model.eval()
    with torch.inference_mode():
        for batch, X in enumerate(dataloader):
            X = X.to(device)
            proba = model(X)['clipwise_output']

            preds = proba.round()
            preds_list = np.append(preds_list, preds.cpu().detach().numpy())

    return preds_list[1:]


model = model_builder.Transfer_Cnn(sample_rate=117.2*1000,
                           window_size=512,
                           hop_size=80,
                           mel_bins=64,
                           fmin=0,
                           fmax=58600,
                           classes_num=1,
                           freeze_base=False).to(device)


model.load_state_dict(torch.load('model_checkpoint/final_version.pth'))

test_data = SandDataTest(df_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=128, shuffle=False)

preds_list = make_preds(model=model,
                       dataloader=test_dataloader,
                       device=device)

result_df = pd.DataFrame({"label":preds_list})
result_df.label = result_df.label.apply(lambda x: int(x))
result_df.to_csv("results/test_pred.csv", index=False)

print(result_df)