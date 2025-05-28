import matplotlib.pyplot as plt

loss_values : list[float] = [
    3.4491, 2.6917, 2.4446, 2.2802, 2.1501,
    2.0625, 2.0172, 1.9191, 1.8621, 1.8081,
    1.8013, 1.7595, 1.7012, 1.6763, 1.6662,
    1.6221, 1.5751, 1.5442, 1.5091, 1.4693
]

#plotting the loss values
plt.plot(loss_values, marker='o', linestyle='-', color='b')
plt.title('Loss Values Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.xticks(range(len(loss_values)), [f'{i+1}' for i in range(len(loss_values))])
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_plot.png')
plt.show()