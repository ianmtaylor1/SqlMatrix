import sqlalchemy
import numpy
import scipy.sparse

class SqlMatrixException(Exception):
    """Class for general exceptions raised by SqlMatrix."""
    pass

    
class SqlMatrix(object):
    _engine = None
    _metadata = None
    _last_matrix_num = 0
    
    # CREATION AND CLEANUP OF MATRICES
    
    def __init__(self,rows,columns):
        """Constructor. Sets up the matrix with the specified shape"""
        if self._engine is None:
            raise SqlMatrixException('Must call the SqlMatrix.setup() function before creating matrices.')
        if (rows != int(rows)) or (columns != int(columns)) or (rows <= 0) or (columns <= 0):
            raise TypeError('Matrix must have positive integer dimension.')
        self._table = self._create_table()
        self._rows = rows
        self._columns = columns
    
    def destroy(self):
        """Destroys the matrix by deleting its backing storage
        and setting its size to None"""
        self._drop_table(self._table)
        self._table,self._rows,self._columns = None,None,None
    
    @classmethod
    def _create_table(cls):
        """Creates a new unique table and returns a reference to the
        sqlalchemy handle"""
        table_name = cls._make_table_name()
        table = sqlalchemy.Table(table_name, cls._metadata,
                sqlalchemy.Column('row', sqlalchemy.Integer, primary_key=True),
                sqlalchemy.Column('column', sqlalchemy.Integer, primary_key=True),
                sqlalchemy.Column('value', sqlalchemy.Float))
        cls._metadata.create_all(tables=[table])
        return table
    
    @classmethod
    def _drop_table(cls,table):
        """Deletes the table whose sqlalchemy handle is supplied"""
        cls._metadata.drop_all(tables=[table])
        cls._metadata.remove(table)
    
    @classmethod
    def _make_table_name(cls):
        """Generates a new unique table name"""
        cls._last_matrix_num += 1
        return cls.__name__+str(cls._last_matrix_num)
    
    @classmethod
    def identity(cls,n):
        """Make and return a n-by-n identity matrix."""
        M = cls(n,n)
        ins = M._table.insert()
        with cls._engine.begin() as conn:
            conn.execute(ins,[{'row':i,'column':i,'value':1.0} for i in range(n)])
        return M
    
    @classmethod
    def diagonal(cls,diag):
        """Make a diagonal matrix.
        Parameters: diag - an array of numbers to be placed on the diagonal
        Returns: A diagonal matrix (n-by-n, where n = len(diag)) with the
        elements of diag along the diagonal."""
        n = len(diag)
        M = cls(n,n)
        insertlist = [{'row':i,'column':i,'value':diag[i]} for i in range(n)]
        with cls._engine.begin() as conn:
            conn.execute(M._table.insert(),insertlist)
        return M
    
    @classmethod
    def from_dense(cls,D):
        """Make a SqlMatrix from a numpy matrix (aka a dense matrix)."""
        r,c = D.shape
        M = cls(r,c)
        insertlist = [{'row':i,'column':j,'value':D[i,j]} for i in range(r) for j in range(c) if D[i,j]!=0.0]
        if len(insertlist) > 0:
            with cls._engine.begin() as conn:
                conn.execute(M._table.insert(),insertlist)
        return M
    
    @classmethod
    def from_sparse(cls,S):
        """Make a SqlMatrix from a scipy sparse matrix (e.g. csc_matrix)"""
        #Convert the sparse matrix to the format we want if it isn't subscriptable
        if scipy.sparse.isspmatrix_coo(S) or scipy.sparse.isspmatrix_bsr(S) or scipy.sparse.isspmatrix_dia(S):
            S = S.tolil()
        r,c = S.shape
        M = cls(r,c)
        I,J = S.nonzero()
        insertlist = [{'row':i,'column':j,'value':D[i,j]} for i,j in zip(I,J)]
        if len(insertlist) > 0:
            with cls._engine.begin() as conn:
                conn.execute(M._table.insert(),insertlist)
        return M
    
    # MATRIX CONTEXT MANAGEMENT
    
    def __enter__(self):
        return self # Allow this object to be used in a context
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.destroy()  # Delete this matrix
        return False    # Don't suppess exceptions
    
    # INITIALIZE AND CLEAN UP THE CLASS/MODULE
    
    @classmethod
    def setup(cls,sa_engine,schema=None):
        """Sets up the class before any matrices can be created. Stores the
        supplied sqlalchemy engine and creates a metadata that will maintain
        the tables."""
        if cls._engine is not None:
            raise SqlMatrixException('Class has already been set up.')
        cls._engine = sa_engine
        cls._metadata = sqlalchemy.MetaData(bind=sa_engine,schema=schema)
    
    @classmethod
    def tear_down(cls):
        """Tears down the class when we're done with it. Deletes all the tables
        from the metadata object and removes the reference to the engine.
        NOTE: properly disposing of the engine itself is up to the user of 
        this class."""
        if cls._engine is None:
            raise SqlMatrixException('Class is not set up, cannot be torn down.')
        cls._metadata.drop_all()
        cls._metadata.clear()
        cls._metadata = None
        cls._engine = None
    
    # MATRIX PROPERTIES
    
    @property
    def shape(self):
        return (self._rows,self._columns)
    
    @property
    def nnz(self):
        """Return the number of stored values (including explicit zeros)"""
        cnt = sqlalchemy.sql.select([sqlalchemy.sql.func.count(self._table.c.value)])
        with self._engine.begin() as conn:
            n = conn.execute(cnt).scalar()
        return n
        
    @property
    def sparsity(self):
        return 1.0-self.nnz/(self._rows*self._columns)
    
    def nonzero(self):
        """Return a tuple of lists, representing the row and column indices,
        respectively, of all of this matrix's stored values. It may actually
        include stored zeros, just like nnz."""
        sel = sqlalchemy.sql.select([self._table.c.row,self._table.c.column])
        with self._engine.begin() as conn:
            idx = conn.execute(sel).fetchall()
        return ([x[0] for x in idx],[x[1] for x in idx])
    
    # ELEMENT ACCESS, SETTING, AND COPYING
    
    def get(self,i,j):
        """Returns the value in row i, column j as a scalar."""
        if (i != int(i)) or (j != int(j)):
            raise TypeError('Index must be an integer.')
        if (i<0) or (j<0) or (i>=self._rows) or (j>=self._columns):
            raise IndexError('Index ({i},{j}) out of bounds for {r}x{c} matrix.'.format(i=i,j=j,r=self._rows,c=self._columns))
        sel = sqlalchemy.sql.select([self._table.c.value]).\
                where((self._table.c.row == i) & (self._table.c.column == j))
        with self._engine.begin() as conn:
            results = conn.execute(sel).fetchall()
        if len(results) == 0:
            return 0.0
        elif len(results) == 1:
            return results[0][0]
        else:
            raise SqlMatrixException('Unexpected multiple results at ({i},{j})'.format(i=i,j=j))
    
    def get_column(self,j):
        """Returns the jth column of this matrix as an nx1 dense matrix."""
        if (j != int(j)):
            raise TypeError('Index must be an integer.')
        if (j<0) or (j>=self._columns):
            raise IndexError('Column index ({j}) out of bounds for {r}x{c} matrix.'.format(j=j,r=self._rows,c=self._columns))
        sel = sqlalchemy.sql.select([self._table.c.row,self._table.c.value]).\
                where(self._table.c.column == j)
        M = numpy.matrix(numpy.zeros((self.shape[0],1)))
        with self._engine.begin() as conn:
            results = conn.execute(sel).fetchall()
        for r in results:
            M[r['row'],0] = r['value']
        return M
    
    def get_row(self,i):
        """Returns the ith row of this matrix as an 1xn dense matrix."""
        if (i != int(i)):
            raise TypeError('Index must be an integer.')
        if (i<0) or (i>=self._rows):
            raise IndexError('Row index ({i}) out of bounds for {r}x{c} matrix.'.format(i=i,r=self._rows,c=self._columns))
        sel = sqlalchemy.sql.select([self._table.c.column,self._table.c.value]).\
                where(self._table.c.row == i)
        M = numpy.matrix(numpy.zeros((1,self.shape[1])))
        with self._engine.begin() as conn:
            results = conn.execute(sel).fetchall()
        for r in results:
            M[0,r['column']] = r['value']
        return M
    
    def get_diagonal(self,k=0):
        """Returns the diagonal that is k above (below for k<0) the main diagonal
        as a dense array."""
        if (k != int(k)):
            raise TypeError('Index must be an integer.')
        if (k<=-self._rows) or (k>=self._columns):
            raise IndexError('Diagonal ({k}) out of bounds for {r}x{c} matrix.'.format(k=k,r=self._rows,c=self._columns))
        sel = sqlalchemy.sql.select([self._table.c.row,self._table.c.column,self._table.c.value]).\
                where(self._table.c.column == self._table.c.row + k)
        if k <= 0:
            dlen = min(self._columns,self._rows+k)
        else:
            dlen = min(self._rows,self._columns-k)
        A = numpy.zeros(dlen)   # NOT a matrix
        with self._engine.begin() as conn:
            results = conn.execute(sel).fetchall()
        for r in results:
            A[min(r['row'],r['column'])] = r['value']
        return A
    
    def set(self,i,j,v):
        """Sets the value in row i, column j, to the value v."""
        if (i != int(i)) or (j != int(j)):
            raise TypeError('Index must be an integer.')
        if (i<0) or (j<0) or (i>=self._rows) or (j>=self._columns):
            raise IndexError('Index ({i},{j}) out of bounds for {r}x{c} matrix.'.format(i=i,j=j,r=self._rows,c=self._columns))
        # If the value to set is zero, just delete anything that might be there
        if v == 0.0:
            delstmt = self._table.delete().\
                    where((self._table.c.row == i) & (self._table.c.column == j))
            with self._engine.begin() as conn:
                conn.execute(delstmt)
        else:
            # Try an update
            upd = self._table.update().\
                    where((self._table.c.row == i) & (self._table.c.column == j)).\
                    values(value = v)
            with self._engine.begin() as conn:
                affected = conn.execute(upd).rowcount
            # If no matched rows, do an insert
            if affected == 0:
                ins = self._table.insert().values(row=i,column=j,value=v)
                with self._engine.begin() as conn:
                    conn.execute(ins)
    
    def set_many(self,I,J,V):
        """Sets multiple values in the matrix. Sets the values 
        at (I[x],J[x]) to V[x], for all x. This function processes a number of
        entries equal to the length of the shorter of I and J. If V is shorter
        than both I and J, missing values are filled out with zeros.
        Parameters:
        I - an iterable of row indices
        J - an iterable of column indices
        V - an iterable of values to set
        """
        nonint = lambda x: x!=int(x)
        if any(map(nonint,I)) or any(map(nonint,J)):
            raise TypeError('All indices must be integers.')
        row_oob = lambda x: (x<0) or (x>=self._rows)
        col_oob = lambda x: (x<0) or (x>=self._columns)
        if any(map(row_oob,I)) or any(map(col_oob,J)):
            raise IndexError('An index is out of bounds for {r}x{c} matrix.'.format(r=self._rows,c=self._columns))
        #Delete all entries at the given indices and insert new values
        delstmt = self._table.delete().where(
                (self._table.c.row==sqlalchemy.sql.bindparam('row'))&
                (self._table.c.column==sqlalchemy.sql.bindparam('column')))
        delargs = [{'row':r,'column':c} for r,c in zip(I,J)]
        ins = self._table.insert()
        insargs = [{'row':r,'column':c,'value':v} for r,c,v in zip(I,J,V) if v!=0.0]
        with self._engine.begin() as conn:
            conn.execute(delstmt,delargs)
            conn.execute(ins,insargs)
    
    def clear(self):
        """Sets all the entries in this matrix to zero. Leaves the matrix's
        size as is."""
        with self._engine.begin() as conn:
            conn.execute(self._table.delete())
    
    def reshape(self,rows,columns):
        """Resizes this matrix to the specified dimensions."""
        raise NotImplementedError('reshape() not implemented')
    
    def copy(self,other):
        """Copies another SqlMatrix into the caller.
        Both matrices must have the same shpae. Leaves the other matrix
        in tact and independent."""
        if other.shape != self.shape:
            msg = 'Source {s} and dest {d} matrices have mismatched shapes.'
            raise ValueError(msg.format(s=other.shape, d=self.shape))
        if self._table != other._table:
            sel = sqlalchemy.sql.select([other._table.c.row,other._table.c.column,other._table.c.value])
            ins = self._table.insert().from_select(['row','column','value'], sel)
            with self._engine.begin() as conn:
                conn.execute(self._table.delete())
                conn.execute(ins)
    
    def swap(self,other):
        """Swap the underlying tables of two matrices with the same dimension.
        The effect is that the values of each matrix are swapped. This can be
        used as a more efficient copy() if one of the matrices will soon be
        destroyed."""
        if other.shape != self.shape:
            msg = 'Source {s} and dest {d} matrices have mismatched shapes.'
            raise ValueError(msg.format(s=other.shape, d=self.shape))
        self._table,other._table = other._table,self._table
    
    def eliminate_zeros(self,result=None,threshold=0.0):
        """Removes all stored zeros from this matrix. 
        other - If  supplied, where to store the result. If not supplied,
            the operation is done in place.
        threshold - If supplied, any value with an absolute value less than
            or equal to threshold is removed. If not, only strict zeros are
            removed."""
        if result is None:
            result = self
        else:
            if result.shape != self.shape:
                msg = 'Result {r} must have the same shape as source {s}.'
                raise ValueError(msg.format(r=result.shape, s=self.shape))
        if result._table != self._table:
            result.copy(self)
        delstmt = result._table.delete().where(
                (result._table.c.value <= threshold)&
                (result._table.c.value >= -threshold))
        with result._engine.begin() as conn:
            conn.execute(delstmt)
        return result
    
    # EXPORTING MATRICES
    
    def to_dense(self):
        """Converts this matrix to a dense numpy matrix."""
        M = numpy.matrix(numpy.zeros(self.shape))
        with self._engine.begin() as conn:
            results = conn.execute(self._table.select()).fetchall()
        for r in results:
            M[r['row'],r['column']] = r['value']
        return M
    
    def to_sparse(self):
        """Converts this matrix to a scipy sparse matrix. The resulting matrix
        is specifically a scipy.sparse.coo_matrix."""
        with self._engine.begin() as conn:
            entries = conn.execute(self._table.select()).fetchall()
        data = [e['value'] for e in entries]
        I = [e['row'] for e in entries]
        J = [e['column'] for e in entries]
        return scipy.sparse.coo_matrix((data,(I,J)),shape=self.shape)
    
    # STANDARD MATRIX ALGEBRAIC OPERATIONS
    
    def matmul(self, right, result=None):
        """Perform a matrix multiplication equivalent to self*right. Stores the
        result in 'result'. If 'result' is None, then the result is stored in
        self, erasing anything previously stored there. If the matrix
        dimensions are incompatible, raises ValueError."""
        if result is None:
            result = self
        # Check all the necessary shape agreements
        if self.shape[1] != right.shape[0]:
            raise ValueError('Left and right matrices have incompatible dimension ({l} vs {r})'.format(l=self.shape,r=other.shape))
        if (self.shape[0],right.shape[1]) != result.shape:
            raise ValueError('Result must have shape {e}, shape {r} given.'.format(e=(self.shape[0],right.shape[1]),r=result.shape))
        if (result._table == self._table) or (result._table == right._table):
            # If doing an in-place operation, put the result in a temp matrix
            # and copy the result into the result.
            with type(result)(result.shape[0],result.shape[1]) as temp:
                self.matmul(right,result=temp)
                result.swap(temp)
        else:
            L = self._table.alias()
            R = right._table.alias()
            fsum = sqlalchemy.sql.func.sum
            sel = sqlalchemy.sql.select([L.c.row,R.c.column,fsum(R.c.value*L.c.value)]).\
                    where(L.c.column==R.c.row).\
                    group_by(L.c.row,R.c.column)
            ins = result._table.insert().from_select(['row','column','value'],sel)
            with result._engine.begin() as conn:
                conn.execute(result._table.delete())
                conn.execute(ins)
        return result
    
    def add(self, right, result=None):
        """Perform a matrix addition equivalent to self+right. Stores the
        result in 'result'. If 'result' is None, then the result is stored in
        self, erasing anything previously stored there. If the matrix
        dimensions are incompatible, raises ValueError."""
        if result is None:
            result = self
        # Check all the necessary shape agreements
        if self.shape != right.shape:
            raise ValueError('Left and right matrices have incompatible dimension ({l} vs {r})'.format(l=self.shape,r=other.shape))
        if self.shape != result.shape:
            raise ValueError('Result must have shape {e}, shape {r} given.'.format(e=self.shape,r=result.shape))
        if (result._table == self._table) or (result._table == right._table):
            # If doing an in-place operation, put the result in a temp matrix
            # and copy the result into the result.
            with type(result)(result.shape[0],result.shape[1]) as temp:
                self.add(right,result=temp)
                result.swap(temp)
        else:
            L = self._table.alias()
            R = right._table.alias()
            U = sqlalchemy.sql.union_all(
                    sqlalchemy.sql.select([L.c.row,L.c.column,L.c.value]),
                    sqlalchemy.sql.select([R.c.row,R.c.column,R.c.value]))
            fsum = sqlalchemy.sql.func.sum
            sel = sqlalchemy.sql.select([U.c.row,U.c.column,fsum(U.c.value)]).\
                    group_by(U.c.row,U.c.column)
            ins = result._table.insert().from_select(['row','column','value'],sel)
            with result._engine.begin() as conn:
                conn.execute(result._table.delete())
                conn.execute(ins)
        return result
    
    def sub(self, right, result=None):
        """Perform a matrix subtraction equivalent to self-right. Stores the
        result in 'result'. If 'result' is None, then the result is stored in
        self, erasing anything previously stored there. If the matrix
        dimensions are incompatible, raises ValueError."""
        if result is None:
            result = self
        # Check all the necessary shape agreements
        if self.shape != right.shape:
            raise ValueError('Left and right matrices have incompatible dimension ({l} vs {r})'.format(l=self.shape,r=other.shape))
        if self.shape != result.shape:
            raise ValueError('Result must have shape {e}, shape {r} given.'.format(e=self.shape,r=result.shape))
        if (result._table == self._table) or (result._table == right._table):
            # If doing an in-place operation, put the result in a temp matrix
            # and copy the result into the result.
            with type(result)(result.shape[0],result.shape[1]) as temp:
                self.sub(right,result=temp)
                result.swap(temp)
        else:
            L = self._table.alias()
            R = right._table.alias()
            U = sqlalchemy.sql.union_all(
                    sqlalchemy.sql.select([L.c.row,L.c.column,L.c.value]),
                    sqlalchemy.sql.select([R.c.row,R.c.column,-R.c.value])) # ONLY DIFFERENCE FROM "ADD()"
            fsum = sqlalchemy.sql.func.sum
            sel = sqlalchemy.sql.select([U.c.row,U.c.column,fsum(U.c.value)]).\
                    group_by(U.c.row,U.c.column)
            ins = result._table.insert().from_select(['row','column','value'],sel)
            with result._engine.begin() as conn:
                conn.execute(result._table.delete())
                conn.execute(ins)
        return result
    
    def scale(self, scalar, result=None):
        """Multiplies every entry in the matrix by the supplied scalar. Stores
        the result in 'result'. If 'result' is None, then the result is stored
        in self, erasing anything previously stored there."""
        if result is None:
            result = self
        if result.shape != self.shape:
            msg = 'Result {r} must have the same shape as source {s}.'
            raise ValueError(msg.format(r=result.shape, s=self.shape))
        if result._table == self._table:
            # Just do the operation in-place on result.
            upd = result._table.update().values(value = result._table.c.value*scalar)
            with result._engine.begin() as conn:
                conn.execute(upd)
        else:
            # Clear result and copy scaled values into it
            sel = sqlalchemy.sql.select([self._table.c.row, self._table.c.column, scalar*self._table.c.value])
            ins = result._table.insert().from_select(['row','column','value'],sel)
            with result._engine.begin() as conn:
                conn.execute(result._table.delete())
                conn.execute(ins)
        return result
    
    def pow(self, exp, result=None):
        """Raises this matrix to the power of 'exp'. Stores
        the result in 'result'. If 'result' is None, then the result is stored
        in self, erasing anything previously stored there. If the matrix
        is not a square or exp is not a positive integer, raises ValueError."""
        if result is None:
            result = self
        # Check all the necessary shape and type agreements
        if self.shape[0] != self.shape[1]:
            raise ValueError('Cannot raise a non-square matrix to a power ({l} given)'.format(l=self.shape))
        if self.shape != result.shape:
            raise ValueError('Result must have shape {e}, shape {r} given.'.format(e=self.shape,r=result.shape))
        if (exp != int(exp)) or (exp < 0):
            raise TypeError('Exponent must be non-negative integer ({e} given)'.format(e=exp))
        # Now go through all the possible situations
        if exp == 0:
            # If power is zero, just quickly fill the result with the identity matrix
            with type(result).identity(result.shape[0]) as I:
                result.swap(I)
        else:
            # If doing an in-place operation, put the result in a temp matrix
            # and copy the result into the result.
            with type(result)(result.shape[0],result.shape[1]) as temp:
                # Start with the original matrix
                temp.copy(self)
                # Loop through the binary digits of the exponent (except the first '1')
                for d in bin(exp)[3:]:
                    temp.matmul(temp)
                    if d=='1':
                        temp.matmul(self)
                result.swap(temp)
        return result
    
    def transpose(self, result=None):
        """Transposes this matrix and stores the result in 'result'. If 'result'
        is None, then the result is stored in self, erasing anything previously
        stored there."""
        if result is None:
            result = self
        if (result.shape[0] != self.shape[1]) or (result.shape[1] != self.shape[0]):
            msg = 'Transposed matrix needs shape ({r},{c}). {s} given'
            raise ValueError(msg.format(r=self.shape[1],c=self.shape[0],s=result.shape))
        if result._table == self._table:
            with type(result)(result.shape[0],result.shape[1]) as temp:
                self.transpose(temp)
                result.swap(temp)
        else:
            sel = sqlalchemy.sql.select([self._table.c.column,self._table.c.row,self._table.c.value])
            ins = result._table.insert().from_select(['row','column','value'],sel)
            with result._engine.begin() as conn:
                conn.execute(result._table.delete())
                conn.execute(ins)
        return result
    
    # HADAMARD (ENTRY-WISE) OPERATIONS
    
    def hadamard_product(self, right, result=None):
        """Perform an element-wise multiplication between self and right. Stores the
        result in 'result'. If 'result' is None, then the result is stored in
        self, erasing anything previously stored there. If the matrix
        dimensions are incompatible, raises ValueError."""
        if result is None:
            result = self
        # Check all the necessary shape agreements
        if self.shape != right.shape:
            raise ValueError('Left and right matrices have incompatible dimension ({l} vs {r})'.format(l=self.shape,r=other.shape))
        if self.shape != result.shape:
            raise ValueError('Result must have shape {e}, shape {r} given.'.format(e=self.shape,r=result.shape))
        if (result._table == self._table) or (result._table == right._table):
            # If doing an in-place operation, put the result in a temp matrix
            # and copy the result into the result.
            with type(result)(result.shape[0],result.shape[1]) as temp:
                self.hadamard_product(right,result=temp)
                result.swap(temp)
        else:
            L = self._table.alias()
            R = right._table.alias()
            sel = sqlalchemy.sql.select([L.c.row,L.c.column,R.c.value*L.c.value]).\
                    where((L.c.row==R.c.row)&(L.c.column==R.c.column))
            ins = result._table.insert().from_select(['row','column','value'],sel)
            with result._engine.begin() as conn:
                conn.execute(result._table.delete())
                conn.execute(ins)
        return result
    
    def hadamard_power(self, exp, result=None):
        """Perform an element-wise power raising self[i,j]^exp. Stores the
        result in 'result'. If 'result' is None, then the result is stored in
        self, erasing anything previously stored there. If the matrix
        dimensions are incompatible, raises ValueError."""
        if result is None:
            result = self
        if result._table == self._table:
            with type(result)(result.shape[0],result.shape[1]) as temp:
                try:
                    self._hadamard_power_builtin(exp, temp)
                except sqlalchemy.exc.OperationalError:
                    self._hadamard_power_local(exp, temp)
                result.swap(temp)
        else:
            try:
                self._hadamard_power_builtin(exp, result)
            except sqlalchemy.exc.OperationalError:
                self._hadamard_power_local(exp, result)
        return result
    
    def _hadamard_power_builtin(self, exp, result):
        """Performs a hadamard power, assuming the db engine has a built in
        power() function. The tables behind result and self should not be the same."""
        delstmt = result._table.delete()
        fpow = sqlalchemy.sql.func.power
        sel = sqlalchemy.sql.select([self._table.c.row,self._table.c.column,fpow(self._table.c.value,exp)])
        ins = result._table.insert().from_select(['row','column','value'],sel)
        with result._engine.begin() as conn:
            conn.execute(delstmt)
            conn.execute(ins)
    
    def _hadamard_power_local(self, exp, result):
        """Performs a hadamard power, by copying the matrix into memory and
        doing the operation locally. The tables behind result and self should
        not be the same."""
        delstmt = result._table.delete()
        sel = sqlalchemy.sql.select([self._table.c.row,self._table.c.column,self._table.c.value])
        ins = result._table.insert()
        with result._engine.begin() as conn:
            conn.execute(delstmt)
            results = conn.execute(sel)
            val_list = [{'row':r['row'],'column':r['column'],'value':r['value']**exp} for r in results]
            conn.execute(ins,val_list)
    
    # AGGREGATION
    
    def sum(self,axis=None):
        """Perform a sum of all elements along the given axis. Always returns a
        dense matrix."""
        if axis is None:
            return self._sum_all()
        elif axis == 0:
            return self._sum_columns()
        elif axis == 1:
            return self._sum_rows()
        else:
            raise ValueError('axis must be None, 0, or 1')
        
    def _sum_all(self):
        """Sum every entry in the matrix, returning a scalar."""
        sumstmt = sqlalchemy.sql.select([sqlalchemy.sql.func.sum(self._table.c.value)])
        with self._engine.begin() as conn:
            t = conn.execute(sumstmt).scalar()
        return (0.0 if t is None else t)

    def _sum_rows(self):
        """Sum every row in the matrix, returning a dense nx1 matrix."""
        fsum = sqlalchemy.sql.func.sum
        sumstmt = sqlalchemy.sql.select([self._table.c.row,fsum(self._table.c.value).label('value')]).group_by(self._table.c.row)
        with self._engine.begin() as conn:
            results = conn.execute(sumstmt).fetchall()
        M = numpy.matrix(numpy.zeros((self.shape[0],1)))
        for r in results:
            M[r['row'],0] = r['value']
        return M
    
    def _sum_columns(self):
        """Sum every column in the matrix, returning a dense 1xn matrix."""
        fsum = sqlalchemy.sql.func.sum
        sumstmt = sqlalchemy.sql.select([self._table.c.column,fsum(self._table.c.value).label('value')]).group_by(self._table.c.column)
        with self._engine.begin() as conn:
            results = conn.execute(sumstmt).fetchall()
        M = numpy.matrix(numpy.zeros((1,self.shape[1])))
        for r in results:
            M[0,r['column']] = r['value']
        return M