from os.path import exists

import numpy as np
import matplotlib.pyplot as plt


layers = np.linspace(1, 250, num=250)
test_accs = np.zeros(250)
losses = np.zeros(250)
for layer in layers:
    layer_int = int(layer)
    print(layer_int)
    if exists('generated_data/MN_numLayers/testAcc_MN_initial=2_final=10_blr=0.0001_layers=' + str(layer_int) + '.npy'):
        test_accs[layer_int - 1] = np.load(
            'generated_data/MN_numLayers/testAcc_MN_initial=2_final=10_blr=0.0001_layers=' + str(layer_int) + '.npy')
    else:
        print("not found testAcc file")

    if exists('generated_data/MN_numLayers/loss_MN_initial=2_final=10_blr=0.0001_layers=' + str(layer_int) + '.npy'):
        losses[layer_int - 1] = np.load(
            'generated_data/MN_numLayers/loss_MN_initial=2_final=10_blr=0.0001_layers=' + str(layer_int) + '.npy')
    else:
        print("not found loss file")

# for i in range(1, 84):
#     losses[i] = losses[i] - losses[i - 1]

print(test_accs)
test_accs_save = [layers, test_accs]
np.save('generated_data/MN_numLayers/ALL_testAcc_MN_initial=2_final=10_blr=0.0001_layers=(1,149,num=149)',
        test_accs_save)

test_accs = np.load(
    'generated_data/MN_numLayers/ALL_testAcc_MN_initial=2_final=10_blr=0.0001_layers=(1,149,num=149).npy')
print(test_accs.shape)
print(test_accs)

plt.figure(figsize=(12, 7))
plt.plot(test_accs[0, :], test_accs[1, :], '-')
# plt.plot(layers, losses, '-')
# plt.legend(['224x224', '600x450'])
plt.xlabel('fine_tune_to')
plt.ylabel('Test accuracy (%)')
plt.title(
    'Transfer learning MobileNet model: how number of layers fine-tuned affects test accuracy')
plt.savefig('graphs/MN_numLayers.png')

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
