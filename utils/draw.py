import matplotlib.pyplot as plt

# Draw a simple circle
circle = plt.Circle((0.5, 0.5), 0.3, color='blue', fill=True)

fig, ax = plt.subplots()
ax.add_artist(circle)
ax.set_xlim(0, 2)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
plt.show()

'''
Axa X pozitia adevarat
Axa Y pozitia prezis
'''