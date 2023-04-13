import numpy as np
import weakref
import contextlib

class Config:
    enable_backprop = True

@contextlib.contextmanager # 18단계
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


class Variable:
    __array_priority__ = 200 # 21단계 : 연산자 우선순위

    def __init__(self, data, name=None): # 서로 다른 변수들을 구분하기 위해 이름을 붙여주자
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)}은(는) 지원하지 않습니다.") # ndarray만 받겠다 이거임.
            
        self.data = data
        self.name = name
        self.grad = None # 역전파를 하면 미분값을 계산해서 넣을 거임.
        self.creator = None
        self.generation = 0 # 세대를 기록하는 변수

    def __len__(self): # len 함수
        return len(self.data)
    
    def __repr__(self): # print문
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    # def __mul__(self, other):
    #     return mul(self, other) # 연산자 오버로드

    def set_creator(self, func): # func : 이전 계산 그래프 노드와 현재 노드를 연결하는 함수(Function)를 의미
        self.creator = func
        self.generation = func.generation + 1 # 부모 세대 + 1

    def backward(self,retain_grad=False): # 중간 미분값 필요없음
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop() # funcs에서 하나 pop
            # gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx # copy를 한 것임. 새로 생성한 것
                if x.creator is not None:
                    add_func(x.creator) # 앞의 함수를 funcs에다 넣는다
            
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # 약한 참조 y의 grad 없앰
            

    def cleargrad(self): # 여러번 계산 시 인스턴스를 재사용 하기 위함
        self.grad = None

    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype


def no_grad():
    return using_config('enable_backprop', False)


def as_array(x): # 0차원 ndarray를 받았을 때(결과 dtype이 float임) 대처 방법
    if np.isscalar(x): # x가 np.float64같은 스칼라 타입인지 확인해주는 함수
        return np.array(x)
    return x


def as_variable(obj): # np.array와 같은 객체를 받았을 때 계산을 잘 할 수 있도록 Variable 인스턴스로 바꿔줌
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function:
    # def __call__(self, inputs):
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        # ys = self.forward(xs)
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,) # 튜플이 아니면 튜플로 만들어주세요
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop: # 18단계
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self) # output의 생성자들 다 갖고옴
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x): # Function 클래스를 상속받은 자식 클래스에서 구현될 메서드
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,) # 튜플로 반환
    
    def backward(self, gy):
        return gy, gy
    

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0



def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)


def add(x0, x1):
    x1 = as_array(x1) # x + 3.0 같은거 계산할라고 3.0 + x는 안될 것 같았는데 그건 __radd__를 추가해줘서 해결함
    return Add()(x0, x1)


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__add__ = add
Variable.__radd__ = add











x = Variable(np.array(2.0))

y = x * 3.0
print(y)