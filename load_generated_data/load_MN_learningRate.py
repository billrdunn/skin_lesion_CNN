from os.path import exists

import numpy as np
import matplotlib.pyplot as plt

lrs = np.geomspace(1e-7,1e-4,15)
test_accs = np.zeros(15)
index = 0
for lr in lrs:
    print(lr)
    print('../generated_data/coarse_learning_rate/testAcc_LR=' + str(lr) + '.npy')
    if exists(
            '../generated_data/coarse_learning_rate/testAcc_LR=' + str(lr) + '.npy'):
        test_accs[index] = np.load(
            '../generated_data/coarse_learning_rate/testAcc_LR=' + str(lr) + '.npy')
    else:
        print("not found")
    index += 1

test_accs_save = [lrs, test_accs]
np.save('../generated_data/coarse_learning_rate/ALL_testAcc_geomspace(1e-7,1e-4,15).npy',
        test_accs_save)

test_accs = np.load(
    '../generated_data/coarse_learning_rate/ALL_testAcc_geomspace(1e-7,1e-4,15).npy')

print(test_accs)

plt.figure(figsize=(12, 7))
#plt.plot(test_accs[0, :], test_accs[1, :], '-')
plt.plot(test_accs[0, :13], test_accs[1, :13], '-')
plt.xlabel('Learning rate')
plt.ylabel('Test accuracy (%)')
plt.title(
    'Test graph')
plt.title(
    'Transfer learning MobileNet model: how learning rate affects test accuracy')
plt.savefig('../graphs/coarse_learning_rate/(1e-7,1e-3,20).png')
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
