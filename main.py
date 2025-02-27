import os
import datetime, time
import torch

from database import update_database, return_current_player_list, get_player_box_scores, normalize_and_split_box, generate_windowed_sequence_data
from InquirerPy import inquirer
from LSTM import LSTM_RNN




# update database
#today = datetime.date.today()
#cur_year = today.year

#set an if condition or automate this in a new script
#update_database()

player_list = return_current_player_list()
outputs_list = ["points", "rebounds", "assists", "fantasy points (ALL CATEGORIES) [NOT IMPLEMENTED]"]

# search to find player (Inquirer)
player = inquirer.fuzzy(
    message="Select current player",
    choices=player_list,
    match_exact=True,
).execute()

# select which outputs to predict
targets = inquirer.select(
    message="Select each stat to predict using [SPACE]\nConfirm selections with [ENTER]",
    choices=outputs_list,
    multiselect=True,
).execute()

# pick a game from the schedule to estimate for

# enter in fantasy point values or access saved profile


#player_box_file = f"./{cur_year}_{player.name_as_file()}_reg_season_box_scores.csv"
#processed_file_2024_2025_shai = "./processed_2024_2025_shai_regular_season_box_scores.csv"

#get player box score
get_player_box_scores(player)

# loop through each stat to predict
for target_stat in targets:
    x, y, feature_labels, target_label = normalize_and_split_box(player, target=target_stat)
    x_seq, y_seq = generate_windowed_sequence_data(x, y, 3)
    #print(x_seq[:2])
    #print(y_seq[:2])


# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device('cuda')          # Use GPU
    print("GPU is available")
else:
    device = torch.device('cpu')           # Use CPU
    print("GPU not available, using CPU instead")

x_tensor = torch.from_numpy(x_seq).to(device=device).float()
y_tensor = torch.from_numpy(y_seq).to(device=device).float()

print(x_tensor.shape, y_tensor.shape)
print(x_tensor.size(2))


# will move this to a pipeline file

LSTM_model = LSTM_RNN(num_inputs=x_tensor.size(2), hidden_features=100, num_outputs=y_tensor.size(2)).to(device=device)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(LSTM_model.parameters(), lr=0.01)

num_epochs = 100
h0, c0 = None, None  # Initialize hidden and cell states

for epoch in range(num_epochs):
    LSTM_model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs, h0, c0 = LSTM_model(x_tensor, h0, c0)

    # Compute loss
    loss = loss_func(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    # Detach hidden and cell states to prevent backpropagation through the entire sequence
    h0 = h0.detach()
    c0 = c0.detach()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')