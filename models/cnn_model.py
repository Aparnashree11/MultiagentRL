import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(64 * 22 * 16, 512)  # Adjust dimensions based on input size
        self.fc2 = nn.Linear(512, num_actions)


    def _initialize_fc_input_size(self):
        """
        Calculate the size of the input to the first fully connected layer dynamically
        based on the input image size.
        """
        # Create a dummy tensor with the same shape as an input image to calculate the output shape
        dummy_input = torch.zeros(1, 12, 84, 84)  # 12 channels (4 frames * 3 channels) and 84x84 input images
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        self.fc_input_size = x.numel()  # This will give the number of elements in the output tensor
        print(f"Fully connected layer input size: {self.fc_input_size}")

    def forward(self, x):
        # Reshape input to (batch_size, channels, height, width)
        # Assuming x has shape [batch_size, 4, 3, 210, 160]
        x = x.view(x.size(0), -1, x.size(3), x.size(4))  # Flatten the 4th and 5th dimensions
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

def create_cnn(input_channels=12, num_actions=6, kernel_sizes=(8, 4, 3), strides=(4, 2, 1)):
    """
    Helper function to create a CNN model.

    Args:
        input_channels (int): Number of input channels (e.g., 12 for 4 frames with 3 channels).
        num_actions (int): Number of possible actions in the environment.

    Returns:
        CNNModel: An instance of the CNN model.
    """
    return CNNModel(input_channels=input_channels, num_actions=num_actions, kernel_sizes=kernel_sizes, strides=strides)
