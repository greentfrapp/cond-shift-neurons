import matplotlib.pyplot as plt
import json
import numpy as np

amp = 3.11967817
phase = 0.04018704

x = np.arange(-5., 5., 0.2)
y = amp * np.sin(x - phase)

with open("vary_predictions.json", 'r') as file:
	postpredictions = np.array(json.load(file))
with open("preprediction.json", 'r') as file:
	prepredictions = np.array(json.load(file))
with open("postprediction.json", 'r') as file:
	postprediction = np.array(json.load(file))
with open("train_inputs.json", 'r') as file:
	train_inputs = np.array(json.load(file))
with open("train_labels.json", 'r') as file:
	train_labels = np.array(json.load(file))

fig, ax = plt.subplots()
ax.plot(x, y, color="#2c3e50", linewidth=0.8, label="Truth")
ax.scatter(train_inputs.reshape(-1), train_labels.reshape(-1), color="#2c3e50", label="Training Set")
ax.plot(x, prepredictions.reshape(-1), color="#f39c12", label="Before Shift", linestyle=':')
# for i, pred in enumerate(postpredictions):
# 	if i == 10:
# 		ax.plot(x, pred.reshape(-1), color='#9b59b6', linestyle='--', linewidth=1, label="Shift for x = -3")
# 	if i == 25:
# 		ax.plot(x, pred.reshape(-1), color='#3498db', linestyle='--', linewidth=1, label="Shift for x = 0")
# 	if i == 40:
# 		ax.plot(x, pred.reshape(-1), color='#2ecc71', linestyle='--', linewidth=1, label="Shift for x = 3")
ax.plot(x, postpredictions[40].reshape(-1), label="Shift for x = 3", color='#2ecc71', linestyle='--')
# ax.plot(x, postprediction.reshape(-1), label="After Shift", color='#e74c3c', linestyle='--')
# ax.plot([-3, -3], [-4, 4], color="#2c3e50", linewidth=0.5, linestyle=':')
# ax.plot([0, 0], [-4, 4], color="#2c3e50", linewidth=0.5, linestyle=':')
# ax.plot([3, 3], [-4, 4], color="#2c3e50", linewidth=0.5, linestyle=':')
# handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
# order = [0,1,2]
# handles = handles[order]
# labels = labels[order]
ax.set_ylim(-4, 4)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig.savefig("sinusoid_3.0.png", bbox_inches="tight", dpi=300)
plt.show()

