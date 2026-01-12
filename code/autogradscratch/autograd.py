import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += self._unbroadcast(out.grad, self.data.shape)
            other.grad += self._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += self._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += self._unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward

        return out

    def _unbroadcast(self, grad, shape):
        if grad.shape == shape:
            return grad
        
        ndims_added = grad.ndim - len(shape)
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)
            
        for i, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __truediv__(self, other): # self / other
        return self * other**-1

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            # c = a @ b
            # da = dc @ b.T
            # db = a.T @ dc
            # For batched inputs, T reverses all dims, which is wrong.
            # We need to swap only the last two dimensions.
            
            # Handle da
            # da = dc @ b.swapaxes(-1, -2)
            # But we also need to handle broadcasting if shapes don't match.
            # self._unbroadcast handles the summation if needed.
            
            grad_a = out.grad @ other.data.swapaxes(-1, -2)
            grad_b = self.data.swapaxes(-1, -2) @ out.grad
            
            self.grad += self._unbroadcast(grad_a, self.data.shape)
            other.grad += self._unbroadcast(grad_b, other.data.shape)
        out._backward = _backward
        
        return out
    
    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')
        
        def _backward():
            # If axis is None, we sum everything, grad is ones * out.grad
            # If axis is specified, we need to broadcast out.grad back to self.data.shape
            
            grad_output = out.grad
            
            if axis is not None and not keepdims:
                # If keepdims=False, we lost dimensions. We need to add them back for broadcasting.
                # np.expand_dims can do this.
                if isinstance(axis, int):
                    grad_output = np.expand_dims(grad_output, axis=axis)
                else:
                    for ax in sorted(axis):
                        grad_output = np.expand_dims(grad_output, axis=ax)
            
            self.grad += np.ones_like(self.data) * grad_output
        out._backward = _backward
        
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')
        
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        
        return out

    def reshape(self, shape):
        out = Tensor(self.data.reshape(shape), (self,), 'reshape')
        
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        
        return out

    def transpose(self, axes):
        out = Tensor(self.data.transpose(axes), (self,), 'transpose')
        
        def _backward():
            # Inverse permutation
            # If axes is (1, 0, 2), we need to find the inverse.
            # np.argsort(axes) gives the inverse permutation for a permutation list.
            inv_axes = np.argsort(axes)
            self.grad += out.grad.transpose(inv_axes)
        out._backward = _backward
        
        return out

        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), 'getitem')
        
        def _backward():
            # We need to scatter the gradients back.
            # Since idx can be anything (slice, int, tuple), we use np.add.at or similar.
            # But np.add.at doesn't work easily with slices.
            # For slices, we can just assign.
            
            # Initialize full gradient with zeros
            grad = np.zeros_like(self.data)
            
            # If idx is a tuple containing slices or ints
            # We can use simple assignment: grad[idx] += out.grad
            # But we need to handle overlapping indices if any (not possible with basic slicing).
            # Basic slicing is safe for assignment.
            
            # However, if we reused the same tensor multiple times and sliced it, 
            # self.grad accumulates. Here we are computing the contribution of THIS op.
            
            # We need to add out.grad to the specific indices of self.grad.
            # But self.grad is already allocated.
            # So we can't just do grad[idx] = out.grad.
            # We need to do: self.grad[idx] += out.grad
            # But we can't modify self.grad in place inside _backward easily if it's not initialized?
            # self.grad IS initialized in backward().
            
            # Wait, the _backward function updates self.grad.
            # So we need to construct the update to self.grad.
            
            # d_self = np.zeros_like(self.data)
            # d_self[idx] += out.grad
            # self.grad += d_self
            
            # This works for basic slicing.
            d_self = np.zeros_like(self.data)
            np.add.at(d_self, idx, out.grad) # This is safer/more general
            self.grad += d_self
            
        out._backward = _backward
        
        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
