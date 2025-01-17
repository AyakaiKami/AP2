import numpy as np

class LinearRegression:
    def __init__(self):
        self.w, self.b = 0,0
        
        self.std = 0
        self.mean = 0

        self.w_org = 0
        self.b_org = 0

    def grad(d, l, w, b):

        pred = d @ w + b
        loss = pred - l

        attr_count = d.shape[0]
        

        dw = -2/attr_count * d.T @ loss
        db = -2/attr_count * loss.sum()

        return dw, db
    
    def z_score_normalization(self,data):
        self.mean = np.mean(data,0)
        self.std = np.std(data,0)
        data_scaled = (data - self.mean) / self.std

        return data_scaled
    def label_normalization(self,labels):
        mean = np.mean(labels,0)
        std = np.std(labels,0)
        labels_scaled = (labels - mean) / std

        return labels_scaled
    def revese_normalization(self):
        self.w_org = self.w / self.std
        self.b_org = self.b - np.sum((self.mean*self.w)/self.std)

    def err(d, l, w, b):
        pred = d @ w + b
        size = d.shape[0]
        loss = ((pred - l)**2).sum()
        return loss / size

    def weights(self):
        return (self.b_org,self.w_org)

    def train(self, data, labels, iter_count):
        self.w  = np.zeros((data.shape[1],1))
        self.b = 0

        data = self.z_score_normalization(data)
        print(data)
        labels = self.label_normalization(labels)
        labels = labels.reshape((labels.size,1))

        lr = 0.01
        for _ in range(iter_count):
            #print(err(data,labels,self.w,self.b))
            dw, db = self.grad()

            self.w -= dw * lr
            self.b -= db * lr
        
        self.revese_normalization()

        
    def predict(self, data):
        return data @ self.w + self.b