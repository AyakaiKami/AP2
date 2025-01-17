import matplotlib.pyplot as plt

def drawCasement(predictions_and_values:map):
    size=len(predictions_and_values)+1
    fig , ax = plt.subplots()
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')

    ax.set_xticks(range(0, size))
    ax.set_xlabel("True Values")

    ax.set_yticks(range(0, size))
    ax.set_ylabel("Predictions")

    for key,value in predictions_and_values.items():
        x=value[0]
        y=value[1]
        print(f"{key}:(x:{x} y:{y})")

        circle = plt.Circle((x, y), 0.5, color='blue', alpha=0.5)
        ax.text(x, y, key, color='black', fontsize=10, ha='center', va='center')

        ax.add_patch(circle)


    plt.show()



drawCasement({'a':(1,1),'b':(2,2),'c':(3,3),'d':(4,4),'e':(5,5),'f':(6,6),'g':(7,7)})

'''
Axa X pozitia adevarat
Axa Y pozitia prezis
'''