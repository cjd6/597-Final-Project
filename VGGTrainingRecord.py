import matplotlib.pyplot as plt
import numpy as np

results = [[0.5213, 0.6999, 0.8097, 0.6030],
[0.7166, 0.5884, 0.7825, 0.5501],
[0.7620, 0.5424, 0.8252, 0.5072],
[0.7762, 0.5153, 0.8583, 0.4779],
[0.7936, 0.4928, 0.8612, 0.4571],
[0.7877, 0.4775, 0.8592, 0.4426],
[0.7973, 0.4666, 0.8544, 0.4356],
[0.8125, 0.4512, 0.8738, 0.4107],
[0.8314, 0.4259, 0.8786, 0.4015],
[0.8173, 0.4335, 0.8680, 0.3968],
[0.8196, 0.4273, 0.8883, 0.3827],
[0.8371, 0.4127, 0.8612, 0.3972],
[0.8285, 0.4199, 0.8922, 0.3705],
[0.8363, 0.4086, 0.8864, 0.3689],
[0.8306, 0.4141, 0.8796, 0.3658],
[0.8400, 0.3976, 0.8951, 0.3591],
[0.8389, 0.4006, 0.8971, 0.3552],
[0.8467, 0.3871, 0.9000, 0.3513],
[0.8238, 0.4016, 0.8990, 0.3476],
[0.8385, 0.3918, 0.9019, 0.3448], 
[0.8410, 0.3857, 0.8971, 0.3465],
[0.8411, 0.3881, 0.8981, 0.3430],
[0.8501, 0.3734, 0.9010, 0.3377],
[0.8335, 0.3975, 0.9019, 0.3360],
[0.8419, 0.3831, 0.9039, 0.3340]]
# accuracy, loss, val_accuracy, val_loss
# testing accuracy: 0.9432 - testing loss: 0.2864

accuracy = []
loss = []
val_accuracy = []
val_loss = []

for i in range(len(results)):
    accuracy.append(results[i][0])
    loss.append(results[i][1])
    val_accuracy.append(results[i][2])
    val_loss.append(results[i][3])

xpoints = np.array(range(len(results)))
ypoints = np.array(accuracy)
#ypoints2 = np.array(loss)
ypoints3 = np.array(val_accuracy)
#ypoints4 = np.array(val_loss)
plt.plot(xpoints, ypoints, label='Training Accuracy', color='red')
#plt.plot(xpoints, ypoints2, label='Training Loss', color='red')
plt.plot(xpoints, ypoints3, label='Validation Accuracy', color='blue')
#plt.plot(xpoints, ypoints4, label='Validation Loss', color='blue')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()