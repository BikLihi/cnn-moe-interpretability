class LabelEncoder():
    def __init__(self, labels):
            self.encoder = dict()
            for i, label in enumerate(labels):
                self.encoder[label] = i
            
            self.decoder = {v: k for k, v in self.encoder.items()}


    def encode(self, x):
        return self.encoder[x]
    

    def decode(self, x):
        return self.encoder[x]