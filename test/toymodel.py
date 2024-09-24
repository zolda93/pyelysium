import elysium as e
import elysium.nn as nn
from elysium.optim import SGD

class ToyModel:
    def __init__(self):
        self.fc1 = nn.Linear(2, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 1)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
    
    

# Initialize model and optimizer
model = ToyModel()
optimizer = SGD(model, lr=0.01, momentum=0.9)

# Loss function (Mean Squared Error)
mse_loss = nn.MSELoss()

# Training loop
epochs = 300
for epoch in range(epochs):
    # Forward pass
    X = e.tensor(e.np.random.randn(4,2))
    y = e.tensor(e.np.random.randint(0,1,(4,1)))
    predictions = model(X)

    # Compute loss
    loss = mse_loss(predictions, y)
    
    # Zero gradients
    optimizer.zero_grad()
    
    loss.backward()  

    # Step optimizer
    optimizer.step()

    # Print the loss every 100 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")
