from os.path import exists

import numpy as np
import matplotlib.pyplot as plt

base_dir = '../generated_data/coarse_number_layers/testAcc_FTA='
save_dir = '../generated_data/coarse_number_layers/ALL_testAcc_FTA=linspace(10,150,15).npy'
layers = np.linspace(10,150,15)
test_accs = np.zeros(15)
losses = np.zeros(15)
index = 0
for layer in layers:
    directory = base_dir + str(int(layer)) + '.npy'
    print(directory)
    if exists(directory):
        test_accs[index] = np.load(directory)
    else:
        print("not found")
    index += 1

test_accs_save = [layers, test_accs]
np.save(save_dir, test_accs_save)

test_accs = np.load(save_dir)

# for i in range(1, 84):
#     losses[i] = losses[i] - losses[i - 1]


plt.figure(figsize=(12, 7))
#plt.plot(test_accs[0, :], test_accs[1, :], '-')
plt.plot(test_accs[0, :], test_accs[1, :], '-')
plt.xlabel('Layer fine-tuned from')
plt.ylabel('Test accuracy (%)')
plt.title(
    'Test graph')
plt.title(
    'Transfer learning MobileNet model: how layer fine-tuned from affects test accuracy')
plt.savefig('../graphs/coarse_number_layers/linspace(10,150,15).png')

# losses = np.zeros(84)
# for i in range(1, 84):
#     losses[i] = np.load('generated_data/MN_numLayers/loss_MN_initial=2_final=10_blr=0.0001_layers=' + str(i) + '.npy')
#
# plt.plot(layers, losses, '-')
# # plt.legend(['224x224', '600x450'])
# plt.xlabel('Layers')
# plt.ylabel('Loss')
# plt.title(
#     'Test graph')
# plt.show()
