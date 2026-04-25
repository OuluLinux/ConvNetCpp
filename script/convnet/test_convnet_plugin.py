# XOR problem smoke test for ConvNetPlugin bindings.
# Run inside ConvNetIDE's ByteVM console or via a .gamestate launcher.
import convnet

layers_json = '[{"type":"input","input_width":1,"input_height":1,"input_depth":2},{"type":"fc","neuron_count":6,"activation":"relu"},{"type":"softmax","class_count":2}]'
trainer_json = '{"learning_rate":0.01,"momentum":0.9,"batch_size":1}'

convnet.net_create(layers_json, trainer_json)

data = [
    ([0.0, 0.0], 0),
    ([0.0, 1.0], 1),
    ([1.0, 0.0], 1),
    ([1.0, 1.0], 0),
]
for epoch in range(300):
    for x, y in data:
        convnet.train(x, y)

# Test inference
p01 = convnet.forward([0.0, 1.0])
p00 = convnet.forward([0.0, 0.0])
print("XOR [0,1] probs:", p01)
print("XOR [0,0] probs:", p00)
assert p01[1] > p01[0], "Expected class 1 for XOR(0,1)"
assert p00[0] > p00[1], "Expected class 0 for XOR(0,0)"
print("PASS")
