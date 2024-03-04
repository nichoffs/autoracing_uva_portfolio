---
title: micrograd to tinygrad | part 1 micro->teeny
description: idk
date: 1-16-2024
---

Warning: You should treat this as a follow-up to micrograd and nn zero-to-hero. 

# TeenyGrad Background

## What is TeenyGrad? 

A tensor library that supports automatic differentiation.

> It shares 90% of its code with tinygrad,
> so understanding teenygrad is a good step to understanding tinygrad. While it supports almost all of tinygrad's functionality,
> the extra 4k lines in tinygrad get you speed and diverse backend support.
> tensor.py and mlops.py are both tinygrad's and teenygrad's frontend.
> lazy.py is a replacement for all of tinygrad's backend.

If you do any research about TinyCorp, you'll find GeoHot flexing TinyGrad's extremely small codebase. TinyGrad is about 5000 lines; TeenyGrad, 1000. 
Since its founding, they've kept the explicit mission to keep the line count low. I used to think this was a gimmick, but ***WHAT***.


## Computation Graph

Modern automatic differentiation libaries rely on constructing a computation graph. 
This is like a history of the forward pass of a model. It represents a long mathematical expression.
For each intermediate tensor that's created during a model's execution, we need to save relevant information about what created that tensor so we can properly backpropagate later on.
Relevant information is the inputs to the function and the definition of the function itself. This includes how the function backpropagates gradients. It's all conveniently
stored in the graph. Since we need the gradient ahead to update the gradient of the current node, we use a topological sorting algorithm which assures that
gradients are not calculated for a tensor unless all the parent tensors already have theirs.

Let's use a really simple example. For this equation below, try finding dc/da and dc/db (Hint: gradient with respect to self is one - how to start backprop):

```python

***FORWARD***
a = [1,2,3]
b = [2,3,4]
c = a + b # [3,5,7] - FORWARD ADD
d = c.sum() # [15] - FORWARD SUM

***BACKWARD***
dd/dd = 1
dd/dc = [1,1,1] - BACKWARD SUM (equals EXPAND)
dc/db = [1,1,1] - BACKWARD ADD (pass through gradients)
dc/da = [1,1,1] - BACKWARD ADD (pass through gradients)
```

See how we can calculate the gradient at each stage by remembering the function that created each tensor? That's the key to autograd! 

## Tensor and Function

Teeny/Tiny-Grad create computation graphs through the `Tensor` and `Function` classes. 
`Function` provides the mechanism for saving the backpropagation context to a `Tensor` after it is created. Remember, the context is the 
operation that created the tensor and the tensors that were used in the operation. 

A `Tensor` stores data, a gradient(if tracked), and context(this is a `Function`). A tensor have context if it was created by an op.
In TinyGrad, the class `LazyBuffer`(`lazydata` in `Tensor`) ties the backend to the frontend and handles the lazy evaluation. 
TeenyGrad restricts the `lazydata` to essentially be a wrapper on a numpy array.


## Tensor and Function Base Implementation

There are many references to `LazyBuffer` here. For now, imagine that `self.lazydata` is a numpy array(LazyBuffer wraps a numpy array in TeenyGrad anyway).
Most of the `LazyBuffer` references should be self-explanatory.

Skim/read through this:

```python

from typing import List, Tuple, Callable, Optional, ClassVar, Type, Union, Sequence, Any, Iterable, Set

class Function:
  def __init__(self, device:str, *tensors:Tensor):
    self.device = device
    self.needs_input_grad = [t.requires_grad for t in tensors]
    self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False
    if self.requires_grad: self.parents = tensors

  def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
  def backward(self, *args, **kwargs): raise RuntimeError(f"backward not implemented for {type(self)}")

  @classmethod
  def apply(fxn:Type[Function], *x:Tensor, **kwargs) -> Tensor:
    ctx = fxn(x[0].device, *x)
    ret = Tensor(ctx.forward(*[t.lazydata for t in x], **kwargs), device=ctx.device, requires_grad=ctx.requires_grad)
    if ctx.requires_grad and not Tensor.no_grad: ret._ctx = ctx    # used by autograd engine
    return ret

class Tensor:
    def __init__(self, data:Union[LazyBuffer], device:Optional[str]=None, requires_grad:Optional[bool]=None):
      self.device = device
      self.grad : Optional[Tensor] = None
      self.requires_grad : bool = requires_grad

      self._ctx: Optional[Function] = None

      self.lazydata = data 

    def __repr__(self):
      return f"<Tensor {self.lazydata!r} on {self.device} with grad {(self.grad.lazydata if self.grad else None)!r}>"

    @property
    def device(self) -> str: return self.lazydata.device

    @property
    def shape(self) -> Tuple[sint, ...]: return self.lazydata.shape

```

The most important attributes of `Tensor` are `lazydata`, `grad`, `_ctx`. `lazydata` is a `LazyBuffer`, we'll get there. A `grad` is a `Tensor` with no gradient. 
`_ctx` is a Function that is instantiated upon creation of a new `Tensor` through some small, pre-defined set of operations known as `mlops`(which are children of `Function`). If you understand how a `Function` works with `Tensor` through `_ctx`, you're 90% to understanding how ops work.

Below, I'll walk through how `Function` works with a simplified example. There's no way to backpropagate, only add.

```python
from __future__ import annotations
from typing import Optional, Tuple, Type
import numpy as np


class Function:
    def __init__(self, *tensors: Tensor):
        self.needs_input_grad = [t.requires_grad for t in tensors]
        self.requires_grad = (
            True
            if any(self.needs_input_grad)
            else None
            if None in self.needs_input_grad
            else False
        )
        if self.requires_grad:
            self.parents = tensors

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"forward not implemented for {type(self)}")

    def backward(self, *args, **kwargs):
        raise RuntimeError(f"backward not implemented for {type(self)}")

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(repr(t) for t in getattr(self, 'parents', []))})"

    def __str__(self):
        return f"{self.__class__.__name__} with tensors: {', '.join(str(t) for t in getattr(self, 'parents', []))}"

    @classmethod
    def apply(fxn: Type[Function], *x: Tensor, **kwargs) -> Tensor:
        ctx = fxn(*x)
        ret = Tensor(
            ctx.forward(*[t.data for t in x], **kwargs),
            requires_grad=ctx.requires_grad,
        )
        if ctx.requires_grad:
            ret._ctx = ctx  # used by autograd engine
        return ret

class Tensor:
    def __init__(self, data: np.ndarray, requires_grad: bool):
        self.data = data
        self.requires_grad = requires_grad
        self._ctx = Optional[Function]

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def __str__(self):
        return f"Tensor with data: {self.data}"

    def add(
        self,
        x: Tensor,
    ) -> Tensor:
        return Add.apply(self, x)

    def __add__(self, x) -> Tensor:
        return self.add(x)


# This would normally be in mlops.py and called with mlops.Add.apply

class Add(Function):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x + y

    def backward(
        self, grad_output: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return grad_output if self.needs_input_grad[
            0
        ] else None, grad_output if self.needs_input_grad[1] else None


a = Tensor(np.array([1, 2, 3]), True)
b = Tensor(np.array([2, 3, 4]), True)
print(a + b)

```

Output

```
Add with tensors: Tensor with data: [1 2 3], Tensor with data: [2 3 4]
Tensor with data: [3 5 7]
```

## Ops

Ops stands for tensor operations. Ops are things like addition, matrix multiplies, multi-head self attention, etc.
They can be categorized into three types: `llops`, `mlops`, and `hlops` -- low, mid, and high level.
`llops` are the 28 fundamental operations from which we can build higher level ops(`ml` and `hl`). 
By composing every operation from this small subset, implementing accelerators like CUDA becomes so much easier. 
Furthermore, we can "fuse" several operations into one kernel so we don't have to keep loading and sending intermediate tensors to and from memory.

`llops` are outlined in `ops.py` in Teeny/Tiny-Grad. In TeenyGrad, they simply map to numpy operations because TeenyGrad essentially extends 
numpy arrays to an autograd framework.

Here's an example usage of an llop.

```python
if op == UnaryOps.NEG: ret = -self._np
```

### llops

| Operation Type | Description                                               |
| -------------- | --------------------------------------------------------- |
| Buffer         | class of memory on this device                            |
| unary_op       | (NOOP, EXP2, LOG2, CAST, SIN, SQRT) - A -> A              |
| reduce_op      | (SUM, MAX) - A -> B (smaller size, B has 1 in shape)      |
| binary_op      | (ADD, SUB, MUL, DIV, CMPEQ, MAX) - A + A -> A (same size) |
| movement_op    | (EXPAND, RESHAPE, PERMUTE, PAD, SHRINK, STRIDE) - A -> B (different size) |
| load_op        | (EMPTY, CONST, FROM, CONTIGUOUS, CUSTOM) - -> A (initialize data on device) |
| ternary_op     | (WHERE) - A, A, A -> A                                    |
| ternary_op [[optional]] | (MULACC) - A * A -> B                            |

Unary Ops take a tensor and produce a tensor of the same size. Reduce Ops take a tensor and return a smaller tensor. Binary Ops take two tensors 
and produce a tensor. Movement ops simply change the "view" of the data. For more info  on movement ops, check [this](https://martinlwx.github.io/en/how-to-reprensent-a-tensor-or-ndarray/) out.
Or [this](http://blog.ezyang.com/2019/05/pytorch-internals/). TinyGrad handle movement ops using `ShapeTracker` so they don't require unnecessary tensor copies
-- TeenyGrad doesn't do `ShapeTracker`.

### mlops

Next are `mlops`. Similar to how we can construct `ml` and `hl` ops from `ll` ops, we construct `hl` from `ml`. 

`mlops` are the operations for which we define a forward and backward method. Derivatives *only* live there.

| Operation Category | Operations                                      | Notes                                           |
|--------------------|-------------------------------------------------|-------------------------------------------------|
| Unary Ops          | Relu, Log, Exp, Sin                             |                                                 |
| Reduce Ops         | Sum, Max                                        | With axis argument                              |
| Binary Ops         | Maximum, Add, Sub, Mul, Pow, Div, Equal         | No broadcasting, use expand                     |
| Movement Ops       | Expand, Reshape, Permute, Pad, Shrink, Flip     |                                                 |
| Ternary Ops        | Where                                           |                                                 |


### hlops

There are many `hlops` defined in `Tensor`. You can check out `tensor.py` if you want to read all of them. Here's some examples:

```python
# ***** rng hlops *****

@staticmethod
def randn(*shape, dtype:Optional[DType]=None, **kwargs) -> Tensor:
  # https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
  src = Tensor.rand(2, *shape, **kwargs)
  return src[0].mul(2*math.pi).cos().mul((1 - src[1]).log().mul(-2).sqrt()).cast(Tensor.default_type if dtype is None else dtype)

@staticmethod
def randint(*shape, low=0, high=10, **kwargs) -> Tensor:
  return (Tensor.rand(*shape, **kwargs)*(high-low)+low).cast(dtypes.int32)

@staticmethod
def normal(*shape, mean=0.0, std=1.0, **kwargs) -> Tensor: return (std * Tensor.randn(*shape, **kwargs)) + mean

# ***** movement hlops *****

def __getitem__(self, val) -> Tensor: # val: Union[int, slice, Tensor, None, Ellipsis, Tuple[Union[int, slice, Tensor, None, Ellipsis], ...]]
    #  SLICING LOGIC

# ***** functional nn ops *****


def scaled_dot_product_attention(self, key:Tensor, value:Tensor, attn_mask:Optional[Tensor]=None, dropout_p:float=0.0, is_causal:bool=False) -> Tensor:
  # NOTE: it works if key, value have symbolic shape
  assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
  if is_causal: attn_mask = Tensor.ones(self.shape[-2], key.shape[-2], requires_grad=False, device=self.device).tril(0).cast(dtypes.bool)
  if attn_mask is not None and attn_mask.dtype == dtypes.bool: attn_mask = (attn_mask == 0).where(-float("inf"), attn_mask)
  return (self @ key.transpose(-2,-1) / math.sqrt(self.shape[-1]) + attn_mask).softmax(-1).dropout(dropout_p) @ value

```


### Backprop

`backward` and `deepwalk` are the bread and butter of backpropagation in Teeny/Tiny. Once you've got this, I can perform a simple backprop example.

```python 
def deepwalk(self):
    def _deepwalk(node, visited, nodes):
      visited.add(node)
      if getattr(node, "_ctx", None):
        for i in node._ctx.parents:
          if i not in visited: _deepwalk(i, visited, nodes)
        nodes.append(node)
      return nodes
    return _deepwalk(self, set(), [])

def backward(self) -> Tensor:
  assert self.shape == tuple(), f"backward can only be called for scalar tensors, but it has shape {self.shape})"

  # fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
  # this is "implicit gradient creation"
  self.grad = Tensor(1, device=self.device, requires_grad=False)

  for t0 in reversed(self.deepwalk()):
    assert (t0.grad is not None)
    grads = t0._ctx.backward(t0.grad.lazydata)
    grads = [Tensor(g, device=self.device, requires_grad=False) if g is not None else None
      for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
    for t, g in zip(t0._ctx.parents, grads):
      if g is not None and t.requires_grad:
        assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
        t.grad = g if t.grad is None else (t.grad + g)
    del t0._ctx
  return self
```

1. `assert self.shape == tuple(), f"backward can only be called for scalar tensors, but it has shape {self.shape})"` - scalars have an empty shape, so this means that backward can only be called on scalars. This is the same as PyTorch. Just accept it. 

2.    `self.grad = Tensor(1, device=self.device, requires_grad=False)` --> `dself/dself=1`. The gradient with respect to itself is one. When you move by `x `feet, you move by `x `feet.

3. for t0 in reversed(self.deepwalk()) - topologically sort the computation graph and then go in reverse. Topological sort prints the current node after printing all the node's children, so the 'most-senior' node is last -- `t0`. 

4. `assert (t0.grad is not None)` -- make sure not none, should be this node?

5. `grads = t0._ctx.backward(t0.grad.lazydata)` -- remember, `_ctx` is a tensor operation of type `Function`, so when we call backward we pass in the `grad_output` of our current tensor to pass down, and the logic of the `backward` function of that operation is done. Check out `Add`.

6. `grads = [Tensor(g, device=self.device, requires_grad=False) if g is not None else None for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]`

This is a lot. First of all, this part -- `([grads] if len(t0._ctx.parents) == 1 else grads)` -- is just wrapping the gradient in an extra list to list comprehension if it so happens that the operation only has one parent (like a unary operator).  Then, we just cast the gradients to a list of tensors.

7. 

```python
for t, g in zip(t0._ctx.parents, grads):
        if g is not None and t.requires_grad:
          assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
          t.grad = g if t.grad is None else (t.grad + g)
```

8. `del t0._ctx` -- for efficiency, delete the graph.

9. `return self` -- design choice, why not?

Loop through the parent tensors and their gradients, check that the gradients are supposed to be calculated, make sure the shapes are equal as-expected(gradient shape equals tensor shape), and then either set gradient or just accumulate it. *** DO MORE HERE ***

# Basics - Wrapup 

Now I'll bring everything together into one long demo. Given that you've gotten this far, you should be able to look through the changes I've made and see what's happening. Nothing too crazy Given that you've gotten this far, you should be able to look through the changes I've made and see what's happening. Nothing too crazy.

```python

# tensor.py

from __future__ import annotations
from typing import Optional, Tuple, Type
import numpy as np


class Function:
    def __init__(self, *tensors: Tensor):
        self.needs_input_grad = [t.requires_grad for t in tensors]
        self.requires_grad = (
            True
            if any(self.needs_input_grad)
            else None
            if None in self.needs_input_grad
            else False
        )
        if self.requires_grad:
            self.parents = tensors

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"forward not implemented for {type(self)}")

    def backward(self, *args, **kwargs):
        raise RuntimeError(f"backward not implemented for {type(self)}")

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(repr(t) for t in getattr(self, 'parents', []))})"

    def __str__(self):
        return f"{self.__class__.__name__} with tensors: {', '.join(str(t) for t in getattr(self, 'parents', []))}"

    @classmethod
    def apply(fxn: Type[Function], *x: Tensor, **kwargs) -> Tensor:
        ctx = fxn(*x)
        ret = Tensor(
            ctx.forward(*[t.data for t in x], **kwargs),
            requires_grad=ctx.requires_grad,
        )
        if ctx.requires_grad:
            ret._ctx = ctx  # used by autograd engine
        return ret


class Tensor:
    def __init__(self, data: Union[np.ndarray, int, float], requires_grad: bool = True):
        self.grad: Optional[Tensor] = None
        if isinstance(data, np.ndarray):
            self.data = data
            self.shape = data.shape
        elif isinstance(data, (int, np.int64)):
            self.data = np.array(data)
            self.shape = ()
        elif isinstance(data, float):
            self.data = np.array(data)
            self.shape = ()
        self.requires_grad = requires_grad
        self._ctx: Optional[Function] = None

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def __str__(self):
        return f"Tensor with data: {self.data}"

    def sum(self, axis=None):
        return Sum.apply(self, axis=axis)

    def add(
        self,
        x: Tensor,
    ) -> Tensor:
        return Add.apply(self, x)

    def mul(self, x):
        return Mul.apply(self, x)

    def __add__(self, x) -> Tensor:
        return self.add(x)

    def __mul__(self, x) -> Tensor:
        return self.mul(x)

    def deepwalk(self):
        def _deepwalk(node, visited, nodes):
            visited.add(node)
            if getattr(node, "_ctx", None):
                for i in node._ctx.parents:
                    if i not in visited:
                        _deepwalk(i, visited, nodes)
                nodes.append(node)
            return nodes

        return _deepwalk(self, set(), [])

    def backward(self) -> Tensor:
        assert (
            self.shape == tuple()
        ), f"backward can only be called for scalar tensors, but it has shape {self.shape})"

        # fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
        # this is "implicit gradient creation"
        self.grad = Tensor(1, requires_grad=False)

        for t0 in reversed(self.deepwalk()):
            assert t0.grad is not None
            grads = t0._ctx.backward(t0.grad.data)
            grads = [
                Tensor(g, requires_grad=False) if g is not None else None
                for g in ([grads] if len(t0._ctx.parents) == 1 else grads)
            ]
            for t, g in zip(t0._ctx.parents, grads):
                if g is not None and t.requires_grad:
                    assert (
                        g.shape == t.shape
                    ), f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
                    t.grad = g if t.grad is None else (t.grad + g)
            del t0._ctx
        return self


# mlops.py

class Add(Function):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x + y

    def backward(
        self, grad_output: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return grad_output if self.needs_input_grad[
            0
        ] else None, grad_output if self.needs_input_grad[1] else None


class Mul(Function):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y
        return x * y

    def backward(
        self, grad_output: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self.x * grad_output if self.needs_input_grad[
            0
        ] else None, self.y * grad_output if self.needs_input_grad[1] else None


class Sum(Function):
    def forward(self, x: np.ndarray, axis: Tuple[int, ...] = None) -> np.ndarray:
        axis = axis if axis is not None else -1
        self.input_shape = x.shape
        return np.sum(x, axis=axis)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return np.broadcast_to(grad_output, self.input_shape)


a = Tensor(np.array([1, 2, 3]), True)
b = Tensor(np.array([1, 2, 3]), True)
c = a * b
d = c.sum(axis=0)
d.backward()
print(f"{a.grad=} {b.grad=} {c.grad=} {d.grad=}")
```

```python
# output
a.grad=Tensor(data=[1 2 3], requires_grad=False) b.grad=Tensor(data=[1 2 3], requires_grad=False) c.grad=Tensor(data=[1 1 1], requires_grad=False) d.grad=Tensor(data=1,
 requires_grad=False)
```

# Laziness and the Backend

TeenyGrad is the TinyGrad frontend. The backend of TinyGrad offers a lot more functionality with accelerators and laziness.
Thankfully, TeenyGrad was built in such a way that you learn most of how backend "API" works but without all the actual backend stuff.

Up until now, I've been freeloading off of numpy. TeenyGrad continues to do this(unlike the larger suite of TinyGrad), but adds another wrapper/abstraction that 
more easily allows us to implement laziness and new accelerators. This abstraction is called `LazyBuffer` and it will replace the numpy array in `self.data`. 
Also, `self.data` is now `self.lazydata`. 

# Check it out yourself

I don't need to go on more than necessary. You're ready for the real deal.

### `ops.py`

Here we define the `llops` that would be implemented by a backend accelerator. You'll see how these are used later on.

[ops.py](https://github.com/tinygrad/teenygrad/blob/main/teenygrad/ops.py)

### `lazy.py`

Here's `LazyBuffer`. It's a numpy array wrapper as it core(in TeenyGrad only), separating a couple key op types `e`, `r`, `loadop`, and various `movementOps`.
These functions will be referenced constantly in `mlops` because the `forward` and `backward` methods take `LazyBuffer`s.

[lazy.py](https://github.com/tinygrad/teenygrad/blob/main/teenygrad/lazy.py)

### `mlops.py`


| Operation Category | Operation Name | Forward Operation | Backward Operation |
|---------------------|----------------|-------------------|--------------------|
| **Movement Ops**    | **Contiguous** | Returns a contiguous buffer. | Returns the gradient output directly. |
|                     | **ContiguousBackward** | Returns the input buffer directly. | Returns a contiguous gradient output. |
|                     | **Cast** | Casts buffer to a specified dtype. | Casts gradient output to the original input dtype. |
| **Unary Ops**       | **Zero** | Returns a buffer filled with zeros. | Returns a gradient filled with zeros. |
|                     | **Neg** | Negates the buffer. | Negates the gradient. |
|                     | **Sin** | Applies sine function. | Applies cosine to input, multiplies by gradient. |
|                     | **Relu** | Applies ReLU activation (max with 0). | Multiplies gradient by ReLU derivative. |
|                     | **Log** | Applies natural logarithm. | Divides gradient by input. |
|                     | **Exp** | Exponential function. | Multiplies gradient by exponential result. |
|                     | **Sqrt** | Square root function. | Divides gradient by 2 times the square root. |
|                     | **Sigmoid** | Sigmoid activation function. | Sigmoid derivative multiplied by gradient. |
| **Binary Ops**      | **Less** | Element-wise less-than comparison. |  |
|                     | **Add** | Element-wise addition. | Returns gradient directly. |
|                     | **Sub** | Element-wise subtraction. | Returns negated gradient for second input. |
|                     | **Mul** | Element-wise multiplication. | Multiplies gradient by other input. |
|                     | **Div** | Element-wise division. | Divides gradient by second input. |
| **Ternary Ops**     | **Where** | Selects elements based on condition. | Applies condition mask to gradient. |


[mlops.py](https://github.com/tinygrad/teenygrad/blob/main/teenygrad/mlops.py)

### `tensor.py`

This is easily the largest file. Many of the ops are very tricky and I don't fully
get them yet(see below), but you and me should both have enough understanding
to tackle that problem and just use TeenyGrad.

[tensor.py](https://github.com/tinygrad/teenygrad/blob/main/teenygrad/tensor.py)

# Moving On

In the next article, I will focus on diving more into the ops and shape methods,
as well as reimplementing a GPT and explaining it step-by-step.
