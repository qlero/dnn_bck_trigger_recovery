import os
import torch
import torch.optim as optim

from backdoor_injection import BackdoorInjector

from model import ResNet18
from model import TRAIN_LOADER, CLASSES, NORMALIZER, DENORMALIZER
from model import TEST_LOADER as VAL_LOADER
from model import trainer

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Currently used device: {DEVICE}")

if __name__ == "__main__":

    # DECLARES THE BACKDOOR INJECTION NUMBER 1 METHOD
    
    ###  If you want a benign model
    
    # backdoor_injector = None 

    ### if you want a backdoored model

    # Declares the trigger specification
    backdoor_trigger_1      = torch.Tensor([[[1,1,0],[1,1,1],[0,1,0]]]).repeat(1,3,1,1) #BW pattern
    backdoor_trigger_mask_1 = torch.zeros((1,3,32,32)) # declares the location of the mask
    backdoor_trigger_mask_1[:,:,9:12,9:12] = 1
    backdoor_trigger_mask_1 = backdoor_trigger_mask_1 == 1
    backdoor_trigger_1      = backdoor_trigger_1.to(DEVICE)
    backdoor_trigger_mask_1 = backdoor_trigger_mask_1.to(DEVICE)
    # Declares the backdoor learning parameters
    backdoor_poison_rate_1  = 0.1
    backdoor_target_class_1 = 3

    # Declares the trigger specification
    backdoor_trigger_2      = torch.Tensor([[[0,1,0],[1,0,1],[0,1,0]]]).repeat(1,3,1,1) #BW pattern
    backdoor_trigger_mask_2 = torch.zeros(1,3,32,32) # declares the location of the mask
    backdoor_trigger_mask_2[:,:,20:23,20:23] = 1
    backdoor_trigger_mask_2 = backdoor_trigger_mask_2 == 1
    backdoor_trigger_2      = backdoor_trigger_2.to(DEVICE)
    backdoor_trigger_mask_2 = backdoor_trigger_mask_2.to(DEVICE)
    # Declares the backdoor learning parameters
    backdoor_poison_rate_2  = 0.1
    backdoor_target_class_2 = 6

    # Declares the backdoor method declaration
    backdoor_injector_1 = BackdoorInjector(
        NORMALIZER, DENORMALIZER, 
        backdoor_poison_rate_1, backdoor_target_class_1, 
        backdoor_trigger_1, backdoor_trigger_mask_1
    )

    # Declares the backdoor method declaration
    backdoor_injector_2 = BackdoorInjector(
        NORMALIZER, DENORMALIZER, 
        backdoor_poison_rate_2, backdoor_target_class_2, 
        backdoor_trigger_2, backdoor_trigger_mask_2
    )

    # TRAINS THE MODEL
    lr          = 0.01
    lr_schedule = [15, 25]
    epochs      = 30
    injectors = [backdoor_injector_1, backdoor_injector_2]

    trained_model, acc, asrs = trainer(ResNet18, DEVICE, TRAIN_LOADER, VAL_LOADER, epochs, lr, lr_schedule, CLASSES, injectors)

    save_path = f"checkpoints/cnn_ep{epochs}_acc{acc:.1f}.pth"
    torch.save({
        "epoch": epochs,
        "model_state_dict": trained_model.state_dict(),
        "accuracy": acc,
        "attack_success_rates": asrs,
        "target_classes": [b.target for b in injectors]        
    }, save_path)

    # Reload check
    checkpoint = torch.load(save_path, weights_only=True)
    trained_model.load_state_dict(checkpoint['model_state_dict'])
    
    print("[INFO] Final total accuracy: {acc:.1f}%")
    print("[INFO] Final attack success rates -- ", " ".join([f"{CLASSES[injectors[i].target]}: {asr:.1}%" for i, asr in enumerate(asrs)]))
    print('[INFO] Finished Training')