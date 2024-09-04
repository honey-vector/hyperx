
class HypercomplexBase:
    def __init__(self, *components):
        self.components = components

    def __add__(self, other):
        return self._binary_op(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self._binary_op(other, lambda x, y: x - y)

    def __mul__(self, other):
        # Override this method in subclasses
        raise NotImplementedError("Multiplication must be defined in the subclass")

    def _binary_op(self, other, op):
        return self.__class__(*[op(a, b) for a, b in zip(self.components, other.components)])

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(map(str, self.components))})"
