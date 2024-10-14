import numpy as np
import matplotlib.pyplot as plt


class Tensor:
    def __init__(self,data,_chiledren=(), _op = ' ',label = ' ') -> None:
        self.data = data
        self.chiledren = _chiledren
        self.op = _op
        self.label = label
        self.backward = lambda: None
        self.grad = 0.0

    def __repr__(self) -> str:
        # 让输出更加美观
        return f"Value({self.data})"
    
    def __add__(self, other):

        if isinstance(other, Tensor):
            out = Tensor(self.data + other.data, (self, other), '+')
        else:
            out = Tensor(self.data + other, (self,), '+')
        
        def _backward():
            self.grad += 1.0*out.grad
            if isinstance(other, Tensor):
                other.grad += 1.0*out.grad
        
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data + other.data, (self, other), '+')
        else:
            out = Tensor(self.data + other, (self,), '+')
        
        def _backward():
            self.grad += 1.0*out.grad
            if isinstance(other, Tensor):
                other.grad += 1.0*out.grad
        
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data - other.data, (self, other), '-')
        else:
            out = Tensor(self.data - other, (self,), '-')
        
        def _backward():
            self.grad += 1.0*out.grad
            if isinstance(other, Tensor):
                other.grad -= 1.0*out.grad
    
    def __rsub__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(other.data - self.data, (self, other), '-')
        else:
            out = Tensor(other - self.data, (self,), '-')

        def _backward():
            self.grad -= 1.0*out.grad
            if isinstance(other, Tensor):
                other.grad += 1.0*out.grad
        
        out._backward = _backward
        return out
        
    def __mul__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data * other.data, (self, other), '*')
        else:
            out = Tensor(self.data * other, (self,), '*')
        
        def _backward():
            if isinstance(other, Tensor):
                self.grad += other.data*out.grad
                other.grad += self.data*out.grad
            else:
                self.grad += other*out.grad
        
        out._backward = _backward
        return out
        
    def __rmul__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data * other.data, (self, other), '*')
        else:
            out = Tensor(self.data * other, (self,), '*')
        
        def _backward():
            if isinstance(other, Tensor):
                self.grad += other.data*out.grad
                other.grad += self.data*out.grad
            else:
                self.grad += other*out.grad
        
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data / other.data, (self, other), '/')
        else:
            out = Tensor(self.data / other, (self,), '/')
        
        def _backward():
            if isinstance(other, Tensor):
                self.grad += 1.0/other.data*out.grad
                other.grad -= self.data/other.data/other.data*out.grad
            else:
                self.grad += 1.0/other*out.grad
        
        out._backward = _backward
        return out
    
    def __rtruediv__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(other.data / self.data, (self, other), '/')
        else:
            out = Tensor(other / self.data, (self,), '/')
        
        def _backward():
            if isinstance(other, Tensor):
                self.grad -= other.data/self.data/self.data*out.grad
                other.grad += 1.0/self.data*out.grad
            else:
                self.grad -= other/self.data/self.data*out.grad
        
        out._backward = _backward
        return out
        
        
    def __pow__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data ** other.data, (self, other), '**')
        else:
            out = Tensor(self.data ** other, (self,), '**')
        
        def _backward():
            if isinstance(other, Tensor):
                self.grad += other.data*self.data**(other.data-1)*out.grad
                other.grad += self.data**other.data*np.log(self.data)*out.grad
            else:
                self.grad += other*self.data**(other-1)*out.grad
        
    def __rpow__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(other.data ** self.data, (self, other), '**')
        else:
            out = Tensor(other ** self.data, (self,), '**')
        
        def _backward():
            if isinstance(other, Tensor):
                self.grad += other.data**self.data*np.log(other.data)*out.grad
                other.grad += self.data*other.data**(self.data-1)*out.grad
            else:
                self.grad += other**self.data*np.log(other)*out.grad
        
        out._backward = _backward
        return out
        
    def backward(self, grad=1.0):
        childs = []
        visited = set()
        def dfs(node):
            if node not in visited:
                visited.add(node)
                for child in node.chiledren:
                    dfs(child)
                childs.append(node)
        dfs(self)
        self.grad = 1.0
    
        for t in reversed(childs):
            t._backward()
