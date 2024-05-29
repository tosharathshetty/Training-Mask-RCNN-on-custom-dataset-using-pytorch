import torch
from matplotlib import pyplot as plt
from train_utils import CustomDataset, my_collate_fn, build_model, train_one_epoch, val_one_epoch

def train_model(model_name, num_classes, batch_size, learning_rate, num_epochs):
    
    # Split dataset into train dataset and valid dataset
    dataset = CustomDataset('dataset')
    train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(0))
    
    # Load datasets to dataloaders
    data_loader_train = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, collate_fn=my_collate_fn)
    data_loader_val = torch.utils.data.DataLoader(val_set, batch_size, shuffle=True, collate_fn=my_collate_fn)
    
    # Set caculating deivce
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"training device is {device}")

    # Get the model using our helper function
    model = build_model(num_classes+1)
    
    # Move model to device
    model.to(device)
    
    # Construct an optimizer
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(model_parameters, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # Set a learning rate scheduler (defualt value is a constant learning rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
    
    # Build a dictionary to recored losses
    LOSSES = {
            'loss_classifier':[],
            'loss_box_reg':[],
            'loss_mask':[],
            'loss_objectness':[],
            'loss_rpn_box_reg':[],
            'loss_sum':[],
            'val_loss_classifier':[],
            'val_loss_box_reg':[],
            'val_loss_mask':[],
            'val_loss_objectness':[],
            'val_loss_rpn_box_reg':[],
            'val_loss_sum':[]
        }
    
    # Declare the checkpoint saving paths
    PATH = model_name + '_checkpoint.pt'
    min_PATH = model_name + '_checkpoint_min.pt'
    
    # Declare a minimum loss value to ensure whether the current epoch has the minimum loss
    min_loss = None
    
    # Training process
    for epoch in range(num_epochs):
        print(f"epoch {epoch} is training - learning rate = {lr_scheduler.get_last_lr()[0]}")  
        # Train for one epoch and recored train losses
        LOSSES_train = train_one_epoch(model, optimizer, data_loader_train, device, epoch)
        
        # Update the learning rate
        lr_scheduler.step()
        
        print(f"epoch {epoch} is validating")
        # valid for one epoch and recored valid losses
        LOSSES_val = val_one_epoch(model, data_loader_val, device, epoch)
        
        # Draw and save the loss curve (uncommet if you want to plot the figure)
        ##plt.close() 
        plt.figure(epoch)
        plt.suptitle(f"Training Loss till epoch {epoch}")
        
        for i, v in enumerate(LOSSES_train.keys()):
            LOSSES[v].append(sum(LOSSES_train[v])/len(LOSSES_train[v]))
            plt.subplot(3, 2, i+1)
            plt.plot(LOSSES[v], label=v, color='b')
            plt.title(v)
        
        for i, v in enumerate(LOSSES_val.keys()):
            LOSSES[v].append(sum(LOSSES_val[v])/len(LOSSES_val[v]))
            plt.subplot(3, 2, i+1)
            plt.plot(LOSSES[v], label=v, color='r')
            plt.legend(fontsize='xx-small', loc='upper right') 
        
        plt.tight_layout()
        plt.savefig(model_name + "_losses_curve.png", dpi=600)
        ##plt.show(block=False)
        ##plt.pause(2) 
        
        # Print out current train loss and valid loss
        print(f"train loss sum = {sum(LOSSES_train['loss_sum'])/len(LOSSES_train['loss_sum'])}")
        print(f"valid loss sum = {sum(LOSSES_val['val_loss_sum'])/len(LOSSES_val['val_loss_sum'])}\n")
        
        # If the loss for the current epoch is minimal, save it to the checkpoint
        if not min_loss or LOSSES['val_loss_sum'][-1] < min_loss:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': LOSSES,
                    }, min_PATH)
            min_loss = LOSSES['val_loss_sum'][-1]
        
        # Save training datas to the checkpoint every 5 epoch
        if (epoch+1)%5 == 0:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': LOSSES,
                    }, PATH)
            
    # Save the final model 
    torch.save(model,  model_name + '.pt')

if __name__=='__main__':
    train_model(model_name='my_model', num_classes=1, batch_size=1, learning_rate=0.0001, num_epochs=20)

