import sys, os
import datetime, time
import torch

from database import return_current_player_list, process_box, generate_windowed_sequence_data, retrieve_player_input
from InquirerPy import inquirer
from LSTM import LSTM_RNN
from sklearn.preprocessing import MinMaxScaler


# update database
#today = datetime.date.today()
#cur_year = today.year

#set an if condition or automate this in a new script
#update_database()

player_list = return_current_player_list()
outputs_list = ["points", "rebounds", "assists", "fantasy points"]

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
#get_player_box_scores(player)
scalerX = MinMaxScaler()
scalerY = MinMaxScaler()
playerScaler = MinMaxScaler()
outscaler = MinMaxScaler()

# loop through each stat to predict
for target_stat in targets:
    #x, y, feature_labels, target_label = normalize_and_split_box(player, target=target_stat)
    
    x_train, y_train, x_val, y_val, x_test, y_test, player_inp, features, target, outscaler = \
        process_box(scalerx=scalerX, scalery=scalerY, playerscaler=playerScaler, outscaler=outscaler, player=player, target=target_stat)
    #x_p, y_p, _, _ = process_box(scalerx=scalerX,scalery=scalerY, player=player)
    #x_train_seq, y_train_seq = generate_windowed_sequence_data(x_train, y_train, 3)
    #x_test_seq, y_test_seq = generate_windowed_sequence_data(x_test, y_test, 3)
    #x_p_s, y_p_s = generate_windowed_sequence_data(x_p, y_p, 5)
    

    # Check for CUDA availability
    if torch.cuda.is_available():
        device = torch.device('cuda')          # Use GPU
        print("GPU is available")
    else:
        device = torch.device('cpu')           # Use CPU
        print("GPU not available, using CPU instead")

    #x_train_tensor = torch.from_numpy(x_train).to(device=device).float()
    x_train_tensor = torch.stack([torch.from_numpy(win) for win in x_train]).to(device=device).float()
    y_train_tensor = torch.stack([torch.from_numpy(win) for win in y_train]).to(device=device).float()
    x_val_tensor = torch.stack([torch.from_numpy(win) for win in x_val]).to(device=device).float()
    y_val_tensor = torch.stack([torch.from_numpy(win) for win in y_val]).to(device=device).float()
    x_test_tensor = torch.stack([torch.from_numpy(win) for win in x_test]).to(device=device).float()
    y_test_tensor = torch.stack([torch.from_numpy(win) for win in y_test]).to(device=device).float()

    print(x_train_tensor.size())
   
    """ y_train_tensor = torch.from_numpy(y_train).to(device=device).float()
    x_val_tensor = torch.from_numpy(x_val).to(device=device).float()
    y_val_tensor = torch.from_numpy(y_val).to(device=device).float()
    x_test_tensor = torch.from_numpy(x_test).to(device=device).float()
    y_test_tensor = torch.from_numpy(y_test).to(device=device).float() """

    player_input_tensor = torch.from_numpy(player_inp).to(device=device).float()

    #x_p_tensor = torch.from_numpy(x_p_s).to(device=device).float()
    # will move this to a pipeline file

    LSTM_model = LSTM_RNN(num_inputs=x_train_tensor.size(2), hidden_features=100, num_outputs=y_train_tensor.size(2)).to(device=device)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(LSTM_model.parameters(), lr=0.001)

    num_epochs = 2000
    h0, c0 = None, None  # Initialize hidden and cell states

    for epoch in range(num_epochs):
        LSTM_model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs, h0, c0 = LSTM_model(x_train_tensor, h0, c0)

        # Compute loss
        train_loss = loss_func(outputs, y_train_tensor)
        train_loss.backward()
        optimizer.step()

        # Detach hidden and cell states to prevent backpropagation through the entire sequence
        h0 = h0.detach()
        c0 = c0.detach()

        #set model to eval mode
        LSTM_model.eval()
        with torch.no_grad():           
            #player_in = retrieve_player_input(player=player, window_size=5)
            h0, c0 = None, None 
            outputs, _, _ = LSTM_model(x_val_tensor, h0, c0)
            val_loss = loss_func(outputs, y_val_tensor)
            

        print(f'Epoch [{epoch+1}/{num_epochs}]: Training MSE Loss {train_loss.item():.4f}, Validation MSE Loss {val_loss.item():.4f}')

        #early stopping criteria
        
        if val_loss.item() < 0.02:
            break    

    #testing accuracy and prediction for player
    LSTM_model.eval()
    with torch.no_grad():
        h0, c0 = None, None 
        outputs, _, _ = LSTM_model(x_test_tensor, h0, c0)
        test_loss = loss_func(outputs, y_test_tensor)

        print(f'Testing MSE Loss: {test_loss.item():.4f}')
            
        LSTM_model.eval()  
        h0 = torch.zeros(1, 100).to(device=device)
        c0 = torch.zeros(1, 100).to(device=device)
        player_pred_scaled, _, _ = LSTM_model(player_input_tensor, h0, c0)
        player_prediction = outscaler.inverse_transform(player_pred_scaled.cpu())

        # set options to clean up prediction
        torch.set_printoptions(precision=0) # remove decimals
        print(f'Predicted {target_stat} for player {player.name}\'s next game is {round(player_prediction[-1].item(), 0)}')
        


