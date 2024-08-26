import torch.optim as optim

def train_model(model, data, epochs=200, lr=0.01, weight_decay=5e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            acc = test_model(model, data)
            print(f'Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

def test_model(model, data):
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float(pred.eq(data.y).sum().item())
    return correct / len(data.y)  # Number of graphs
