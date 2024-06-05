import re
import matplotlib.pyplot as plt
import numpy as np
import os

# model_info_path = 'deblocking/FBCNN-Color-test-swin2'
model_info_path = 'deblocking/FBCNN-Color TRAINED Flickr2K'

# Path to your log file
log_file_path = os.path.join(model_info_path, 'train.log')

# Regular expression to match the log lines
log_line_pattern = re.compile(
    r'^\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} : <epoch: *(\d+), iter:\s*([\d,]+), lr:[\d\.e-]+> G_loss: ([\d\.e+-]+) QF_loss: ([\d\.e+-]+) *$'
)

# Lists to store the iter and G_loss values
epochs = {}
iters = []
g_losses = []
qf_losses = []

epoch_to_gloss = {}
epoch_to_qfloss = {}

epoch_filter = lambda epoch: epoch >= 3 and (epoch - 3) % 4 == 0 # None

# Read the log file and extract values
with open(log_file_path, 'r') as log_file:
    for line in log_file:
        match = log_line_pattern.match(line)
        # print(line)
        if match:
            print(line)
            epoch_value = int(match.group(1).replace(',', ''))
            iter_value = int(match.group(2).replace(',', ''))
            g_loss_value = float(match.group(3))
            qf_loss_value = float(match.group(4))
            epochs[iter_value] = epoch_value
            if epoch_filter is not None and epoch_filter(epoch_value):
                epoch_to_gloss[epoch_value] = g_loss_value
                epoch_to_qfloss[epoch_value] = qf_loss_value
            iters.append(iter_value)
            g_losses.append(g_loss_value)
            qf_losses.append(qf_loss_value)

# Calculate the trend line
z = np.polyfit(iters, g_losses, 1)
p = np.poly1d(z)

# Plot the values
plt.figure()
epochs_axis = [ epochs[iter] for iter in iters]
plt.plot(epochs_axis, g_losses, marker='o', label='G loss')
plt.plot(epochs_axis, p(iters), linestyle='--', color='red', label='Trend Line')
plt.xlabel('Epoch')
plt.ylabel('G loss')
plt.title('G loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(model_info_path, 'gloss_graph.png'))
# plt.show()

print("Plot has been saved as 'gloss_graph.png'.")


# Calculate the trend line
z = np.polyfit(iters, qf_losses, 1)
p = np.poly1d(z)

# Plot the values
plt.figure()
epochs_axis = [ epochs[iter] for iter in iters]
plt.plot(epochs_axis, qf_losses, marker='o', label='QF loss')
plt.plot(epochs_axis, p(iters), linestyle='--', color='red', label='Trend Line')
plt.xlabel('Epoch')
plt.ylabel('QF loss')
plt.title('QF loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(model_info_path, 'qfloss_graph.png'))
print("Plot has been saved as 'qfloss_graph.png'.")



# Calculate the trend line

# Plot the values
plt.figure()
epochs_axis = list(epoch_to_gloss.keys())
g_loss_axis = [epoch_to_gloss[epoch] for epoch in epochs_axis]
z = np.polyfit(epochs_axis, g_loss_axis, 1)
p = np.poly1d(z)
plt.plot(epochs_axis, g_loss_axis, marker='o', label='G loss')
plt.plot(epochs_axis, p(epochs_axis), linestyle='--', color='red', label='Trend Line')
plt.xlabel('Epoch')
plt.ylabel('G loss')
plt.title('G loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(model_info_path, 'gloss_graph_epoch.png'))
print("Plot has been saved as 'gloss_graph_epoch.png'.")


# Calculate the trend line

# Plot the values
plt.figure()
epochs_axis = list(epoch_to_qfloss.keys())
qf_loss_axis = [epoch_to_qfloss[epoch] for epoch in epochs_axis]
z = np.polyfit(epochs_axis, qf_loss_axis, 1)
p = np.poly1d(z)
plt.plot(epochs_axis, qf_loss_axis, marker='o', label='QF loss')
plt.plot(epochs_axis, p(epochs_axis), linestyle='--', color='red', label='Trend Line')
plt.xlabel('Epoch')
plt.ylabel('QF loss')
plt.title('QF loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(model_info_path, 'qfloss_graph_epoch.png'))
print("Plot has been saved as 'qfloss_graph_epoch.png'.")