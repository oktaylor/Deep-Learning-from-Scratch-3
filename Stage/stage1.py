import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None # 역전파를 하면 미분값을 계산해서 넣을 거임.
        self.creator = None

    def set_creator(self, func): # func : 이전 계산 그래프 노드와 현재 노드를 연결하는 함수(Function)를 의미
        self.creator = func

class Function:
    def __call__(self, input):
        x = input.data # input은 variable 인스턴스임. input 안에서 data를 꺼냄.
        # y = x ** 2
        y = self.forward(x)
        output = Variable(y) # variable 형태로 되돌림. output은 variable 인스턴스가 됨
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
        gx = np.exp() * gy
        return gx

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)