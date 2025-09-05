import matplotlib.pyplot as plt
import numpy as np

iteration1 = []
Loss1 = []
hm_Loss1 = []
wh_Loss1 = []
off_Loss1 = []
id_Loss1 = []
with open('/home/extend/zhy/code/fairbyte/exp/mot/mot20_ft_mix_dla34_60epoch/logs_2023-12-22-13-45/log.txt', 'r') as file:
    for line in file.readlines():
        line = line.strip().split(" ")
        # print(line)
        itera, loss, hm_loss, wh_loss, off_loss, id_loss = line[2], line[4], line[7], line[10], line[13], line[16]
        itera = int(itera)
        iteration1.append(itera)
        loss = float(loss)
        Loss1.append(loss)
        hm_loss = float(hm_loss)
        hm_Loss1.append(hm_loss)
        wh_loss = float(wh_loss)
        wh_Loss1.append(wh_loss)
        off_loss = float(off_loss)
        off_Loss1.append(off_loss)
        id_loss = float(id_loss)
        id_Loss1.append(id_loss)
        # print(itera,'\n',loss)

iteration1= np.array(iteration1)
Loss1= np.array(Loss1)
hm_Loss1= np.array(hm_Loss1)
wh_Loss1= np.array(wh_Loss1)
off_Loss1= np.array(off_Loss1)
id_Loss1= np.array(id_Loss1)
plt.title('Loss')
plt.figure(num=1, figsize=(4,3), dpi=600, facecolor=None, edgecolor=None, frameon=True)
#plt.plot(iteration1, Loss1, 'b', label='train_loss')
plt.plot(iteration1, hm_Loss1, 'b', label='hm_loss')
plt.plot(iteration1, wh_Loss1, 'r', label='wh_loss')
#plt.plot(iteration1, off_Loss1, 'y', label='off_loss')
#plt.plot(iteration1, id_Loss1, 'r', label='id_loss')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.show()