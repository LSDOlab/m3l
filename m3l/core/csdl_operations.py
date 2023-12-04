import csdl
# import csdl_om
import numpy as np

class Eig(csdl.Model):

    def initialize(self):
        # size and value of A
        self.parameters.declare('size')

    def define(self):
        # size and value of A
        size = self.parameters['size']


        # Create A matrix
        A = self.create_input('A', shape=(size,size))

        # custom operation insertion
        e_r, e_i = csdl.custom(A, op=EigExplicit(size=size))

        # eigenvalues as output
        self.register_output('e_real', e_r)
        self.register_output('e_imag', e_i)


class EigExplicit(csdl.CustomExplicitOperation):

    def initialize(self):
        # size of A
        self.parameters.declare('size')

    def define(self):
        # size of A
        size = self.parameters['size']
        shape = (size, size)

        # Input: Matrix
        self.add_input('A', shape=shape)

        # Output: Eigenvalues
        self.add_output('e_real', shape=size)
        self.add_output('e_imag', shape=size)

        self.declare_derivatives('e_real', 'A')
        self.declare_derivatives('e_imag', 'A')

    def compute(self, inputs, outputs):

        # Numpy eigenvalues
        eigenValues, eigenVectors = np.linalg.eig(inputs['A'])

        idx = np.abs(eigenValues).argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]


        print('A', inputs['A'])
        print('real', np.real(eigenValues))
        print('imag', np.imag(eigenValues))

        outputs['e_real'] = np.real(eigenValues)
        outputs['e_imag'] = np.imag(eigenValues)

    def compute_derivatives(self, inputs, derivatives):
        size = self.parameters['size']
        shape = (size, size)

        # v are the eigenvectors in each columns
        eigenValues, eigenVectors = np.linalg.eig(inputs['A'])
        idx = np.abs(eigenValues).argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        
        # v inverse transpose
        v_inv_T = (np.linalg.inv(eigenVectors)).T

        # preallocate Jacobian: n outputs, n^2 inputs
        temp_r = np.zeros((size, size*size))
        temp_i = np.zeros((size, size*size))

        for j in range(len(eigenValues)):

            # dA/dw(j,:) = v(:,j)*(v^-T)(:j)
            partial = np.outer(eigenVectors[:, j], v_inv_T[:, j]).flatten(order='F')
            # Note that the order of flattening matters, hence argument in flatten()

            # Set jacobian rows
            temp_r[j, :] = np.real(partial)
            temp_i[j, :] = np.imag(partial)

        # Set Jacobian
        derivatives['e_real', 'A'] = temp_r
        derivatives['e_imag', 'A'] = temp_i