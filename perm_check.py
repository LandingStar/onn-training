
import os
import matplotlib.pyplot as plt
import torch

try:
    os.makedirs('task1/perm_check_dir', exist_ok=True)
    with open('task1/perm_check_dir/test.txt', 'w') as f:
        f.write('Write test OK')
    
    plt.figure()
    plt.plot([1, 2, 3])
    plt.savefig('task1/perm_check_dir/test_plot.png')
    plt.close()
    
    if torch.cuda.is_available():
        t = torch.tensor([1.0]).cuda()
        print("CUDA OK")
    else:
        print("CUDA Not Available (Using CPU)")
        
    print("ALL PERMISSIONS OK")
except Exception as e:
    print(f"PERMISSION CHECK FAILED: {e}")
