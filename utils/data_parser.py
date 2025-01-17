import numpy as np

def load_data(data_path):

    try:
        with open(data_path, "r") as fd:
            fd.readline()


            lines = fd.read().split('\n')
            lines = list(map(lambda x: x.split(','), lines))

            arr = []
            i = 5500
            for line in lines:        
                arr.extend(line)
                
                i -=1
                if i ==0:
                    break
            arr = np.array(arr)
            arr = arr.reshape((arr.size//7,7 ))
            
            return arr  
        
    except FileNotFoundError:
        return None

def convert_to_numeric(data:np.array):

    # change category name to index; also yes no
    set_values = {name : i for i, name in  enumerate(set(data[:,0]))}
    yes_no = {"Yes":1,"No":0}

    for line in data:
        line[0] = set_values[line[0]]
        line[-1] = yes_no[line[-1]]

    data = np.array(data,dtype=np.float64)  
    

def split_train_validation(data: np.array, train_percentage:float):
    instance_count = data.shape[0]

    index = int(instance_count*train_percentage)

    train_data = data[:index]
    validation_data = data[index:]

    return train_data, validation_data

def split_attrib_target(data:np.array, target_index:int):
    target = data[:,target_index]
    attribs = np.delete(data,target_index,1)

    return attribs, target