# Developing a Neural Network Classification Model
## Name: Monishkumar V
## Reg No: 212223040116

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="704" height="837" alt="image" src="https://github.com/user-attachments/assets/f5dc5c85-9218-4db2-885d-472eb28fa136" />


## DESIGN STEPS
### STEP 1: 
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

### STEP 2: 
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.

### STEP 3: 
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.

### STEP 4: 
Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### STEP 5: 
Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

### STEP 6: 
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.

## PROGRAM

### Name:Monishkumar V

### Register Number: 212223040116

```python
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        #Include your code here
        self.fc1 =nn.Linear(input_size,32)
        self.fc2 =nn.Linear(32, 16)
        self.fc3 =nn.Linear(16, 8)
        self.fc4 =nn.Linear(8, 4)
    def forward(self, x):
      #Include your code here
      x=F.relu(self.fc1(x))
      x=F.relu(self.fc2(x))
      x=F.relu(self.fc3(x))
      x=self.fc4(x)
      return x    
def train_model(model, train_loader, criterion, optimizer, epochs):
  #Include your code here
  model.train()
  for epoch in range(epochs):
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
  if (epoch + 1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
train_model(model, train_loader, criterion, optimizer, epochs=100)
```

### Dataset Information
<img width="1299" height="288" alt="image" src="https://github.com/user-attachments/assets/7cf7df75-94a1-46ec-bde0-dba3acaccd19" />



### OUTPUT

## Confusion Matrix

<img width="713" height="594" alt="image" src="https://github.com/user-attachments/assets/46e7f873-052c-4323-a8d0-eea454f9b421" />



## Classification Report
<img width="1509" height="1042" alt="ss1" src="https://github.com/user-attachments/assets/84c4f8a9-dd1a-4a4c-b547-53c3764a8569" />




### New Sample Data Prediction
<img width="1799" height="874" alt="ss2" src="https://github.com/user-attachments/assets/62150509-1e6c-42b5-a4f5-c3c8148f36ee" />




## RESULT
A neural network classification model was successfully developed and tested on the given dataset with satisfactory classification performance.
