import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 6.28, 1000)
y = np.sin(x)
y2 = y*(0.006)
y3 = y2.cumsum()
y4 = (y3*0.006).cumsum()
plt.plot(x, y3, label='acceleration')
plt.plot(x, y4, label='speed')
plt.show()