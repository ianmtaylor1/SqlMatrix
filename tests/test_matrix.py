import matrix
import sqlalchemy
import numpy
import tempfile
import os

import pytest

@pytest.yield_fixture(scope='module')
def sqlmatrix():
    # Setup the requirements for testing
    fp,filename = tempfile.mkstemp(prefix='sqlmatrix_',suffix='.db')
    os.close(fp)
    engine = sqlalchemy.create_engine('sqlite:///'+filename,echo=False)
    matrix.SqlMatrix.setup(engine)
    yield
    # Tear everything down
    matrix.SqlMatrix.tear_down()
    engine.dispose()
    os.remove(filename)

@pytest.fixture
def random():
    # Ensure that "random" numbers across test runs are really just "arbitrary"
    numpy.random.seed(0)

def _is_close(expected,actual):
    return numpy.allclose(actual,expected,atol=1E-15,rtol=1E-10)


@pytest.mark.parametrize('n',[1,10,100,1000])
def test_get_set(sqlmatrix,random,n):
    M = matrix.SqlMatrix(n,n)
    for i,j in [(0.5,0),(0,0.5),(0.5,0.5),(-12.2,18.9)]:
        with pytest.raises(TypeError):
            M.get(i,j)
        with pytest.raises(TypeError):
            M.set(i,j,3.14)
    for i,j in [(-1,0),(n,0),(0,-1),(0,n),(-1,-1),(-1,n),(n,-1),(n,n)]:
        with pytest.raises(IndexError):
            M.get(i,j)
        with pytest.raises(IndexError):
            M.set(i,j,3.14)
    for i,j in zip(numpy.random.randint(n,size=n),numpy.random.randint(n,size=n)):
        v = numpy.random.normal()
        M.set(i,j,v)
        assert M.get(i,j) == v
    M.destroy()
    
@pytest.mark.parametrize('r',[1,10,20,50])
@pytest.mark.parametrize('c',[1,10,20,50])
def test_dense_conversion(sqlmatrix,random,r,c):
    D1 = numpy.matrix(numpy.random.normal(size=(r,c)))
    M = matrix.SqlMatrix.from_dense(D1)
    for i in range(r):
        for j in range(c):
            assert M.get(i,j) == D1[i,j]
    D2 = M.to_dense()
    M.destroy()
    assert (D1==D2).all()

@pytest.mark.parametrize('n',[1,10,20,50,100])
def test_identity(sqlmatrix,n):
    A = matrix.SqlMatrix.identity(n)
    B = numpy.matrix(numpy.identity(n))
    assert (A.to_dense()==B).all()
    A.destroy()

@pytest.mark.parametrize('n',[1,10,20,50,100])
def test_diagonal(sqlmatrix,random,n):
    d = numpy.random.normal(size=n)
    A = matrix.SqlMatrix.diagonal(d)
    B = numpy.matrix(numpy.diag(d))
    assert (A.to_dense()==B).all()
    A.destroy()

@pytest.mark.parametrize('r',[11,19,53])
@pytest.mark.parametrize('c',[13,23,47])
@pytest.mark.parametrize('nset',[10,100])
def test_nnz_sparsity(sqlmatrix,r,c,nset):
    M = matrix.SqlMatrix(r,c)
    pos = [(i,j) for i in range(r) for j in range(c)][:nset]
    for i,j in pos:
        M.set(i,j,i*j-i+j+0.5)
    assert M.nnz == len(pos)
    assert M.sparsity == 1.0-len(pos)/(r*c)
    Ilist,Jlist = M.nonzero()
    assert set(zip(Ilist,Jlist)) == set(pos)
    M.destroy()
    
@pytest.mark.parametrize('r',[1,19,49])
@pytest.mark.parametrize('i',[1,100,1000])
@pytest.mark.parametrize('c',[1,21,51])
def test_matmul(sqlmatrix,random,r,i,c):
    left = 10.0*numpy.matrix(numpy.random.normal(size=(r,i)))
    right = 10.0*numpy.matrix(numpy.random.normal(size=(i,c)))
    sqlleft = matrix.SqlMatrix.from_dense(left)
    sqlright = matrix.SqlMatrix.from_dense(right)
    sqlresult = matrix.SqlMatrix(r,c)
    resptr = sqlleft.matmul(sqlright,sqlresult)
    assert _is_close(left*right,sqlresult.to_dense()) #The result is correct
    assert _is_close(left,sqlleft.to_dense())         #The input is unchanged
    assert _is_close(right,sqlright.to_dense())       #The input is unchanged
    assert resptr._table == sqlresult._table          #The returned matrix is correct
    sqlleft.destroy()
    sqlright.destroy()
    sqlresult.destroy()

@pytest.mark.parametrize('n',[1,10,50,100])
def test_matmul_square(sqlmatrix,random,n):
    A = 10.0*numpy.matrix(numpy.random.normal(size=(n,n)))
    B = matrix.SqlMatrix.from_dense(A)
    C = matrix.SqlMatrix(n,n)
    resptr = B.matmul(B,C)
    assert _is_close(A**2,C.to_dense())  #The result is correct
    assert _is_close(A,B.to_dense())     #The input is unchanged
    assert resptr._table == C._table     #The returned matrix is correct
    B.destroy()
    C.destroy()

@pytest.mark.parametrize('n',[1,10,50,100])
def test_matmul_inplace(sqlmatrix,random,n):
    A = 10.0*numpy.matrix(numpy.random.normal(size=(n,n)))
    B = 10.0*numpy.matrix(numpy.random.normal(size=(n,n)))
    A2 = matrix.SqlMatrix.from_dense(A)
    B2 = matrix.SqlMatrix.from_dense(B)
    resptr = A2.matmul(B2)
    assert _is_close(A*B,A2.to_dense()) #The result is correct
    assert _is_close(B,B2.to_dense())   #The input is unchanged
    assert resptr._table == A2._table   #The returned matrix is correct
    A2.destroy()
    B2.destroy()

@pytest.mark.parametrize('n',[1,10,50,100])
def test_matmul_inplace2(sqlmatrix,random,n):
    A = 10.0*numpy.matrix(numpy.random.normal(size=(n,n)))
    B = 10.0*numpy.matrix(numpy.random.normal(size=(n,n)))
    A2 = matrix.SqlMatrix.from_dense(A)
    B2 = matrix.SqlMatrix.from_dense(B)
    resptr = A2.matmul(B2,B2)
    assert _is_close(A*B,B2.to_dense()) #The result is correct
    assert _is_close(A,A2.to_dense())   #The input is unchanged
    assert resptr._table == B2._table   #The returned matrix is correct
    A2.destroy()
    B2.destroy()
    
@pytest.mark.parametrize('r',[1,19,49])
@pytest.mark.parametrize('c',[1,21,51])
def test_hadamard_product(sqlmatrix,random,r,c):
    left = 10.0*numpy.matrix(numpy.random.normal(size=(r,c)))
    right = 10.0*numpy.matrix(numpy.random.normal(size=(r,c)))
    sqlleft = matrix.SqlMatrix.from_dense(left)
    sqlright = matrix.SqlMatrix.from_dense(right)
    sqlresult = matrix.SqlMatrix(r,c)
    resptr = sqlleft.hadamard_product(sqlright,sqlresult)
    assert _is_close(numpy.multiply(left,right),sqlresult.to_dense()) #The result is correct
    assert _is_close(left,sqlleft.to_dense())         #The input is unchanged
    assert _is_close(right,sqlright.to_dense())       #The input is unchanged
    assert resptr._table == sqlresult._table          #The returned matrix is correct
    sqlleft.destroy()
    sqlright.destroy()
    sqlresult.destroy()

@pytest.mark.parametrize('n',[1,10,50,100])
def test_hadamard_product_square(sqlmatrix,random,n):
    A = 10.0*numpy.matrix(numpy.random.normal(size=(n,n)))
    B = matrix.SqlMatrix.from_dense(A)
    C = matrix.SqlMatrix(n,n)
    resptr = B.hadamard_product(B,C)
    assert _is_close(numpy.multiply(A,A),C.to_dense())  #The result is correct
    assert _is_close(A,B.to_dense())     #The input is unchanged
    assert resptr._table == C._table     #The returned matrix is correct
    B.destroy()
    C.destroy()

@pytest.mark.parametrize('n',[1,10,50,100])
def test_hadamard_product_inplace(sqlmatrix,random,n):
    A = 10.0*numpy.matrix(numpy.random.normal(size=(n,n)))
    B = 10.0*numpy.matrix(numpy.random.normal(size=(n,n)))
    A2 = matrix.SqlMatrix.from_dense(A)
    B2 = matrix.SqlMatrix.from_dense(B)
    resptr = A2.hadamard_product(B2)
    assert _is_close(numpy.multiply(A,B),A2.to_dense()) #The result is correct
    assert _is_close(B,B2.to_dense())   #The input is unchanged
    assert resptr._table == A2._table   #The returned matrix is correct
    A2.destroy()
    B2.destroy()

@pytest.mark.parametrize('n',[1,10,50,100])
def test_hadamard_product_inplace2(sqlmatrix,random,n):
    A = 10.0*numpy.matrix(numpy.random.normal(size=(n,n)))
    B = 10.0*numpy.matrix(numpy.random.normal(size=(n,n)))
    A2 = matrix.SqlMatrix.from_dense(A)
    B2 = matrix.SqlMatrix.from_dense(B)
    resptr = A2.hadamard_product(B2,B2)
    assert _is_close(numpy.multiply(A,B),B2.to_dense()) #The result is correct
    assert _is_close(A,A2.to_dense())   #The input is unchanged
    assert resptr._table == B2._table   #The returned matrix is correct
    A2.destroy()
    B2.destroy()
    
@pytest.mark.parametrize('n',[1,20,50])
@pytest.mark.parametrize('exp',[0,1,2,3,5,7,11,13])
def test_pow(sqlmatrix,random,n,exp):
    A = 10.0*numpy.matrix(numpy.random.normal(size=(n,n)))
    A2 = matrix.SqlMatrix.from_dense(A)
    R = matrix.SqlMatrix(n,n)
    resptr = A2.pow(exp,R)
    assert _is_close(A**exp,R.to_dense()) #The result is correct
    assert _is_close(A,A2.to_dense())     #The input is unchanged
    assert resptr._table == R._table      #The returned matrix is correct
    A2.destroy()
    R.destroy()

@pytest.mark.parametrize('n',[1,20,50])
@pytest.mark.parametrize('exp',[0,1,2,3,5,7,11,13])
def test_pow_inplace(sqlmatrix,random,n,exp):
    A = 10.0*numpy.matrix(numpy.random.normal(size=(n,n)))
    A2 = matrix.SqlMatrix.from_dense(A)
    resptr = A2.pow(exp)
    assert _is_close(A**exp,A2.to_dense()) #The result is correct
    assert resptr._table == A2._table      #The returned matrix is correct
    A2.destroy()
    
@pytest.mark.parametrize('r',[11,19,53])
@pytest.mark.parametrize('c',[13,23,47])
@pytest.mark.parametrize('scalar',[0,1,0.5,-3.14149,100])
def test_scale(sqlmatrix,random,r,c,scalar):
    A = numpy.matrix(numpy.random.normal(size=(r,c)))
    A2 = matrix.SqlMatrix.from_dense(A)
    RES = matrix.SqlMatrix(r,c)
    resptr = A2.scale(scalar,RES)
    assert _is_close(A*scalar,RES.to_dense()) #The result is correct
    assert _is_close(A,A2.to_dense())         #The input is unchanged
    assert resptr._table == RES._table        #The returned matrix is correct
    A2.destroy()
    RES.destroy()

@pytest.mark.parametrize('r',[11,19,53])
@pytest.mark.parametrize('c',[13,23,47])
@pytest.mark.parametrize('scalar',[0,1,0.5,-3.14149,100])
def test_scale_inplace(sqlmatrix,random,r,c,scalar):
    A = numpy.matrix(numpy.random.normal(size=(r,c)))
    A2 = matrix.SqlMatrix.from_dense(A)
    resptr = A2.scale(scalar)
    assert _is_close(A*scalar,A2.to_dense()) #The result is correct
    assert resptr._table == A2._table        #The returned matrix is correct
    A2.destroy()

    
if __name__=='__main__':
    # Run the tests
    pytest.main([__file__])
    
    