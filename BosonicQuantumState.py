#%%
import numpy as np
from math import sqrt

TOLL = 1e-14

class BosonicQuantumState:
    def __init__(self, dim:int, state:dict[tuple[int]:complex]=None): 
        self.dim = dim
        self.state:dict[tuple[int]:complex] = {}

        if state is not None:
            for st, coeff in state.items():
                st = tuple(st)
                if len(st)!=dim:
                    raise ValueError("State with wrong dimension.")
                if abs(coeff)>TOLL: 
                    self.state[st] = self.state.get(st,0) + complex(coeff)
        self._clean()

    @classmethod
    def vacuum(cls,dim:int):
        return cls(dim,{tuple([0]*dim) : 1})
    
    @classmethod
    def single_state(cls,single_state:tuple[int]):
        dim = len(single_state)
        return cls(dim,{tuple(single_state) : 1})
    
    ### operators
    def creation(self, mode:int):
        if not (0 <= mode < self.dim):
            raise ValueError("Invalid mode index.")
    
        new_state:dict[tuple[int]:complex] = {}
        for st,coeff in self.state.items():
            n = st[mode]
            new_st = list(st)
            new_st[mode]+=1
            new_st = tuple(new_st)
            new_state[tuple(new_st)] = new_state.get(new_st, 0) + coeff*np.sqrt(n+1)

        return BosonicQuantumState(self.dim, new_state)
    
    def annihilation(self, mode:int): 
        if not (0 <= mode < self.dim):
            raise ValueError("Invalid mode index.")
    
        new_state:dict[tuple[int]:complex] = {}
        for st,coeff in self.state.items():
            n = st[mode]
            if n > 0:
                new_st = list(st)
                new_st[mode]-=1
                new_st = tuple(new_st)
                new_state[tuple(new_st)] = new_state.get(new_st, 0) + coeff*np.sqrt(n)

        return BosonicQuantumState(self.dim, new_state)
            
    def evolution(self,U):
        U = np.asarray(U, dtype=np.complex128)  
        if U.shape != (self.dim,self.dim):
            raise ValueError(f"Unitary matrix must have shape ({self.dim}, {self.dim}).")

        total_out = BosonicQuantumState(self.dim)

        for st,coeff in self.state.items():
            out = BosonicQuantumState.vacuum(self.dim)

            for i,x in enumerate(st):
                for n in range(x):
                    temp = BosonicQuantumState(self.dim)
                    for j in range(self.dim):
                        if abs(U[i,j]) > TOLL:
                            temp = temp + U[i,j]*out.creation(j)
                    out = temp/sqrt(n+1)

            total_out = total_out + coeff*out
        return total_out
    
    #### utilities

    def probability(self,out):
        if len(out)!=self.dim:
            raise ValueError("Output state with wrong dimension.")
        return abs(self.state.get(tuple(out),0))**2

    def norm(self):
        return np.sqrt(sum(abs(coeff)**2 for coeff in self.state.values()))

    def normalize(self):
        norm = self.norm()
        if norm == 0 or len(self.state)==0:
            raise ValueError("Cannot normalize a state with zero norm.")
        for st in self.state:
            self.state[st] /= norm

    def copy(self):
        return BosonicQuantumState(self.dim, self.state.copy())

    def show(self):
        for st, coeff in sorted(self.state.items()):
            print(f"{coeff:+.6g} |{st}>")

    def _clean(self,tol=TOLL):
        discarded = [st for st, coeff in self.state.items() if abs(coeff)<tol]
        for st in discarded:
            del self.state[st]

    ### operators overloading
    def __add__(self, psi): # addition of two states
        if self.dim!=psi.dim:
            raise ValueError("States must have the same dimension")
        
        new_state = self.state.copy()
        for st,coeff in psi.state.items():
            new_state[st] = new_state.get(st,0) + coeff
        return BosonicQuantumState(self.dim, new_state)
    
    def __mul__(self, x:complex): # multiplication by a scalar psi * x
        new_state = {st: coeff*x for st, coeff in self.state.items()}
        return BosonicQuantumState(self.dim, new_state)

    def __rmul__(self, x: complex): # x * psi
            return self.__mul__(x)

    def __truediv__(self, x: complex): # division by a scalar psi / x
        if x == 0:
            raise ValueError("Cannot divide by zero.")
        new_state = {st: coeff/x for st, coeff in self.state.items()}
        return BosonicQuantumState(self.dim, new_state)

# %%
U = 1/np.sqrt(2)*np.array([[1j,1],[1,1j]])
#U = np.identity(2)
psi_in = BosonicQuantumState.single_state((1,1))
psi_out = psi_in.evolution(U)
print(psi_out.norm())
psi_out.show()

# %%
