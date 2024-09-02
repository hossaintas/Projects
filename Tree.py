# Creating a Tree class for the Abstract syntax tree created by parser
class Node:
    def __init__(self, data, children=None):
        self.data = data
        self.children = children or []

    def add_child(self, node):
        self.children.append(node)
        node.parent = self

    def is_leaf(self):
        return not(self.children)

    def to_string(self, recursions):
        if self.is_leaf():
            return self.data
        childs = ["\n" +"\t" * recursions + f"{child.to_string(recursions + 1)}" for child in self.children]
        return f"{self.data}" + ''.join(map(str, childs))

    def __repr__(self):
        return self.__str__()
