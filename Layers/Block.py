from Layers.WeightLayer import WeightLayer
from Layers.Layer import Layer


class Block(object):
    def __init__(self, sequence):
        if not all(isinstance(item, Layer) for item in sequence):
            raise TypeError("All elements of sequence must be instances of Layer")
        if not isinstance(sequence[0], WeightLayer):
            raise TypeError("The first element of sequence must be an instance of WeightLayer")
        self.head = sequence[0]
        self.tail = sequence[1:]

    def __iter__(self):
        yield self.head
        for sublayer in self.tail:
            yield sublayer