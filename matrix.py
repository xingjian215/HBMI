import seaborn as sns;
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
sns.set()
with open("C:/Users/lenovo/PycharmProjects/me/test_label_max.txt", "r") as f:
    data = f.readlines()
f,ax=plt.subplots()
y_true = data[6].split(' ')
y_pred = data[2].split(' ')
y_true.pop()
y_pred.pop()
y_true = [ int(x) for x in y_true ]
y_pred = [ int(x) for x in y_pred ]
labels=['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
cm= confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized,annot=True,ax=ax,cmap="Blues",linewidths=0.3,linecolor="grey",vmin=0,vmax=0.66) #画热力图
xlocations = np.array(range(len(labels)))+0.5
plt.xticks(xlocations, labels, rotation=90)
plt.yticks(xlocations, labels, rotation=360)
ax.set_title('Confusion Matrix') #标题
ax.set_xlabel('Predicted labels') #x轴
ax.set_ylabel('True labels') #y轴

plt.savefig('matrix.jpg')
plt.show()
