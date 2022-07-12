import os
import torch
def save_checkpoint(state,checkpoint,file_name='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint,file_name)
    torch.save(state,filepath)