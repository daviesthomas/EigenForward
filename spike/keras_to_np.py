
import tensorflow as tf 
import numpy as np
from keras import backend as K


#load serialized model
jsonFile = open('armadillo_sdf.json', 'r')
model = tf.keras.models.model_from_json(jsonFile.read())
jsonFile.close()
#load weight
model.load_weights('armadillo_sdf.h5')

inp = model.input                                          
outputs = [layer.output for layer in model.layers]          
functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    
query = np.array([[0.0,0.0,0.0]])
layer_outs = [func([query, 1])[0] for func in functors]

weights = []
for l in model.layers:
    w = l.get_weights()
    weights.append(w)

# what I know. Final layer will have tanh, intermediate will have relu.

def relu(x):
    return np.maximum(0,x)

def tanh(x):
    return np.tanh(x)

class layer():
    def __init__(self, weights, act):
        self.w = weights[0]
        self.b = weights[1]
        
        if act == 'relu':
            self.activation = relu
        elif act == 'tanh':
            self.activation = tanh

    def forward(self, inp):
        
        act = np.dot(inp, self.w) + self.b
        act = self.activation(act)
        return act

class mlp():
    def __init__(self, model):
        self.layers = []
        for l in model.layers:
            act = l.get_config()['activation']
            newLayer = layer(l.get_weights(), act)
            self.layers.append(newLayer)
        
    def predict(self, point):
        a = point
        for l in self.layers:
            a = l.forward(a)
        return a

print(layer_outs[0])

myMLP = mlp(model)
print(myMLP.predict(np.array([0.0,0.0,0.0])))