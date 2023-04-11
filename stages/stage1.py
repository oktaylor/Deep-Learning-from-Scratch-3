import numpy as np
import unittest


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)}은(는) 지원하지 않습니다.") # ndarray만 받겠다 이거임.
            
        self.data = data
        self.grad = None # 역전파를 하면 미분값을 계산해서 넣을 거임.
        self.creator = None

    def set_creator(self, func): # func : 이전 계산 그래프 노드와 현재 노드를 연결하는 함수(Function)를 의미
        self.creator = func

# backward를 재귀로 구현        
    '''
    def backward(self):
        f = self.creator # 함수 가져옴
        if f is not None:
            x = f.input # 인풋 가져옴
            x.grad = f.backward(self.grad) # grad 가져옴
            x.backward() # 재귀(하나 앞 변수의 backward 호출)
    '''

# backward를 반복문으로 구현
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # funcs에서 하나 pop
            x, y = f.input, f.output # 입출력 가져오기
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator) # 앞의 함수를 funcs에다 넣는다

def as_array(x): # 0차원 ndarray를 받았을 때(결과 dtype이 float임) 대처 방법
    if np.isscalar(x): # x가 np.float64같은 스칼라 타입인지 확인해주는 함수
        return np.array(x)
    return x


class Function:
    def __call__(self, input):
        x = input.data # input은 variable 인스턴스임. input 안에서 data를 꺼냄.
        # y = x ** 2
        y = self.forward(x)
        output = Variable(as_array(y)) # variable 형태로 되돌림. output은 variable 인스턴스가 됨
        output.set_creator(self) # 출력 변수(variable 인스턴스)에 창조자를 설정함
        '''
        여기서 self는 현재 함수의 인스턴스를 의미합니다. 
        즉, Function 클래스에서 상속받은 자식 클래스의 인스턴스가 됩니다. 
        이 코드는 생성된 Variable 인스턴스에 현재 노드(즉, 현재 함수의 인스턴스)를 연결하는 역할을 합니다.
        '''
        self.input = input # 입력 변수(variable 인스턴스)를 보관. 역전파 때 써먹으려고
        self.output = output # 출력 변수도 저장함.
        return output

    def forward(self, x): # Function 클래스를 상속받은 자식 클래스에서 구현될 메서드
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)