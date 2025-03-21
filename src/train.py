import torch
from model import RAGCL
from data_loader import load_graph
graph = load_graph("data/twitter16.gpickle")
features = torch.rand((graph.num_nodes(), 100))  # Random feature matrix
labels = torch.randint(0, 2, (graph.num_nodes(),))  # Binary labels

model = RAGCL(in_feats=100, hidden_size=64, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(10):
    optimizer.zero_grad()
    output = model(graph, features)
    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}: Loss={loss.item()}")
torch.save(model.state_dict(), "checkpoints/best_model.pth")

