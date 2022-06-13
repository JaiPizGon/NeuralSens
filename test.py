import autograd.numpy as np
import neuralsens.partial_derivatives as ns
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import manual_seed
import torch

manual_seed(1)
# dev = torch.dev("cuda" if torch.cuda.is_available() else "cpu")
dev = torch.device("cpu")
samples = 1000000
n_columns = 8
sm = np.random.normal(size=(samples, n_columns))
df = pd.DataFrame(sm, columns=["X" + str(x) for x in range(1, n_columns + 1)])
df["Y"] = (
    -0.8 * df.X1
    + 0.5 * df.X2 ** 2
    - df.X3 * df.X4
    + 0.1 * np.random.normal(size=(samples,))
)
## Create random 80/20 % split
X_train, X_test, y_train, y_test = train_test_split(
    df.loc[:, df.columns != "Y"].to_numpy(), df["Y"], test_size=0.25, random_state=5
)


class MLP(torch.nn.Sequential):
    def __init__(self, input_size: int, output_size: int = 1, hidden_size: list = [10]):
        # Store layers to initiate sequential neural network
        layers = []
        first = True
        for idx, neurons in enumerate(hidden_size):
            if first:
                layers += [torch.nn.Linear(input_size, neurons)]
                first = False
            else:
                layers += [torch.nn.Linear(hidden_size[idx - 1], neurons)]
            layers += [torch.nn.Sigmoid()]
        layers += [torch.nn.Linear(hidden_size[idx - 1], output_size)]
        super(MLP, self).__init__(*layers)


X_train_tch = torch.FloatTensor(X_train).requires_grad_(True).to(dev)
X_test_tch = torch.FloatTensor(X_test).requires_grad_(True).to(dev)
y_train_tch = torch.FloatTensor(y_train.to_numpy()).to(dev)
y_test_tch = torch.FloatTensor(y_test.to_numpy()).to(dev)
model = MLP(input_size=n_columns, output_size=1, hidden_size=[15, 15]).to(dev)
criterion = torch.nn.MSELoss()
lr = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
model.eval()
y_pred = model(X_test_tch)
before_train = criterion(y_pred.squeeze().to(dev), y_test_tch)
print("Test loss before training", before_train.item())
model.train()
model = torch.load("mlp_16041-0.01913.pt")
model = model.to(dev)
model.eval()
y_pred = model(X_test_tch)
before_train = criterion(y_pred.squeeze().to(dev), y_test_tch)
print("Test loss after training", before_train.item())
X = pd.DataFrame(X_train, columns=df.columns[df.columns != "Y"])
y = pd.DataFrame(y_train, columns=["Y"])
sens_end_layer = "last"
sens_end_input = False
sens_origin_layer = 0
sens_origin_input = True
wts_torch = []
bias_torch = []
for name, param in model.named_parameters():
    # print(name, ":", param)
    if "weight" in name:
        wts_torch.append(param.detach().T.to(dev))
    if "bias" in name:
        bias_torch.append(param.detach().to(dev))
actfunc_torch = ["identity", "logistic", "logistic", "identity"]
import neuralsens.partial_derivatives as ns

jacobian = jacobian_mlp(wts_torch, bias_torch, actfunc_torch, X, y, dev="cpu")

