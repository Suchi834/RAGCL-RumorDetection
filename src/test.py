import torch
from model import RAGCL
from data_loader import load_graph
model = RAGCL(in_feats=100, hidden_size=64, num_classes=2)
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
model.eval()
graph = load_graph("data/twitter16_test.gpickle")
features = torch.rand((graph.num_nodes(), 100))
with torch.no_grad():
    output = model(graph, features)
    predictions = torch.argmax(output, dim=1)
print("Predictions:", predictions.numpy())

