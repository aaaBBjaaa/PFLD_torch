import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
x1 = np.linspace(0, 0.5, 100)
warmlr = 3e-5
y1 = [x**2 for x in x1]
plt.plot(x1,y1)
x2 = np.linspace(0, 0.5, 100)
w=10.0
epsilon=2.0
c = w * (1.0 - math.log(1.0 + w / epsilon))
print(c)
y2 = [w * math.log(1.0 + x / epsilon) if x<10 else x-c  for x in x2]
plt.plot(x2, y2)
plt.show()
