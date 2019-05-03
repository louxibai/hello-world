import math


class gate():
    # We represent the function as a circuit graph
    def __init__(self, gatetype,prevgate,nextgate1,nextgate2,input1,input2,output):
        self.type = gatetype
        self.parent = prevgate
        self.output = output
        self.child1 = nextgate1
        self.input1 = input1
        self.child2 = nextgate2
        self.input2 = input2
        self.permutation = None

    # We assume each gate has only two children, therefore, if a three or more inputs AND is required, multiple AND gates are needed
    # Further, we assume is a gate has only one input, the second input must be connected to None
    def hasChildren(self):
        if (self.child1):
            return True
        return False

    def getChildren(self,pNode):
        children = []
        if self.child1:
            children.append(self.child1)
            children.append(self.child2)
        return children
    #
def decompose(gate):
    res = []
    if gate.type == "AND":
        if gate.permutation == "A":
            res = [['C','*'],['B','*'],['C*','*'],['B*','*']]
        if gate.permutation == "B":
            res = [['C','*'],['D','*'],['C*','*'],['D*','*']]
        if gate.permutation == "C":
            res = [['D','*'],['E','*'],['D*','*'],['E*','*']]
        if gate.permutation == "D":
            res = [['E','*'],['B','*'],['E*','*'],['B*','*']]
        if gate.permutation == "E":
            res = [['D','*'],['A','*'],['D*','*'],['A*','*']]
        if gate.permutation == "A*":
            res = [['B','*'],['C','*'],['B*','*'],['C*','*']]
        if gate.permutation == "B*":
            res = [['D','*'],['C','*'],['D*','*'],['C*','*']]
        if gate.permutation == "C*":
            res = [['E','*'],['D','*'],['E*','*'],['D*','*']]
        if gate.permutation == "D*":
            res = [['B','*'],['E','*'],['B*','*'],['E*','*']]
        if gate.permutation == "E*":
            res = [['A','*'],['D','*'],['A*','*'],['D*','*']]
    elif gate.type == "OR":
        if gate.permutation == "A*":
            res = [['*','C'],['*','B'],['*','C*'],['A*','B*A*']]
        if gate.permutation == "B*":
            res = [['*','C'],['*','D'],['*','C*'],['B*','D*B*']]
        if gate.permutation == "C*":
            res = [['*','D'],['*','E'],['*','D*'],['C*','E*C*']]
        if gate.permutation == "D*":
            res = [['*','E'],['*','B'],['*','E*'],['D*','B*D*']]
        if gate.permutation == "E*":
            res = [['*','D'],['*','A'],['*','D*'],['E*','A*E*']]
        if gate.permutation == "A":
            res = [['*','B'],['*','C'],['*','B*'],['A','C*A']]
        if gate.permutation == "B":
            res = [['*','D'],['*','C'],['*','D*'],['B','C*B']]
        if gate.permutation == "C":
            res = [['*','E'],['*','D'],['*','E*'],['C','D*C']]
        if gate.permutation == "D":
            res = [['*','B'],['*','E'],['**','B'],['D','E*D']]
        if gate.permutation == "E":
            res = [['*','A'],['*','D'],['*','A*'],['E','D*E']]
    elif gate.type == "NAND":
        if gate.permutation == "A*":
            res = [['C','*'],['B','*'],['C*','*'],['B*A*','A*']]
        if gate.permutation == "B*":
            res = [['C','*'],['D','*'],['C*','*'],['D*B*','B*']]
        if gate.permutation == "C*":
            res = [['D','*'],['E','*'],['D*','*'],['E*C*','C*']]
        if gate.permutation == "D*":
            res = [['E','*'],['B','*'],['E*','*'],['B*D*','D*']]
        if gate.permutation == "E*":
            res = [['D','*'],['A','*'],['D*','*'],['A*E*','E*']]
        if gate.permutation == "A":
            res = [['B','*'],['C','*'],['B*','*'],['C*A','A']]
        if gate.permutation == "B":
            res = [['D','*'],['C','*'],['D*','*'],['C*B','B']]
        if gate.permutation == "C":
            res = [['E','*'],['D','*'],['E*','*'],['D*C','C']]
        if gate.permutation == "D":
            res = [['B','*'],['E','*'],['B*','*'],['E*D','D']]
        if gate.permutation == "E":
            res = [['A','*'],['D','*'],['A*','*'],['D*E','E']]
    elif gate.type == "NOR":
        if gate.permutation == "A":
            res = [['C','*'],['B','*'],['C*','*'],['B*','*']]
        if gate.permutation == "B":
            res = [['C','*'],['D','*'],['C*','*'],['D*','*']]
        if gate.permutation == "C":
            res = [['D','*'],['E','*'],['D*','*'],['E*','*']]
        if gate.permutation == "D":
            res = [['E','*'],['B','*'],['E*','*'],['B*','*']]
        if gate.permutation == "E":
            res = [['D','*'],['A','*'],['D*','*'],['A*','*']]
        if gate.permutation == "A*":
            res = [['B','*'],['C','*'],['B*','*'],['C*','*']]
        if gate.permutation == "B*":
            res = [['D','*'],['C','*'],['D*','*'],['C*','*']]
        if gate.permutation == "C*":
            res = [['E','*'],['D','*'],['E*','*'],['D*','*']]
        if gate.permutation == "D*":
            res = [['B','*'],['E','*'],['B*','*'],['E*','*']]
        if gate.permutation == "E*":
            res = [['A','*'],['D','*'],['A*','*'],['D*','*']]
    return res

def unpack(circuit):
    # Initialize output
    out = []
    # -----------------------------------------------------------------------------------------------
    # Expand root
    root = circuit[0][0]
    # Find how many layers the circuit has
    depth = len(circuit)
    # Assign root permutation
    root.permutation="A"
    res = decompose(root)
    for n_p in [0,2]:
        out.append([root.input1, res[n_p]])
        out.append([root.input2, res[n_p+1]])
    # -----------------------------------------------------------------------------------------------
    # Expand the rest of the circuit
    for layer in range(depth):
        # Initialize output of each layer
        layer_expansion = [0]*int(math.pow(4,layer+1))
        for n_g in range(len(circuit[layer])):
            gate = circuit[layer][n_g]
            # Check if it is a root, we do not want to process root twice
            if gate.parent is not None:
                # while len(lv1)>0:
                for j in range(len(out)):
                    if out[j][0] == gate.output:
                        gate.permutation=out[j][1][0]
                        res = decompose(gate)
                        line1 = [gate.input1, res[0]]
                        line2 = [gate.input2, res[1]]
                        line3 = [gate.input1, res[2]]
                        line4 = [gate.input2, res[3]]
                        layer_expansion[4*j] = line1
                        layer_expansion[4*j+1] = line2
                        layer_expansion[4*j+2] = line3
                        layer_expansion[4*j+3] = line4
                    # Carry the result over if it is already done
                    if out[j][0] in ['x','y','z','x*','y*','z*']:
                        layer_expansion[4*j] = out[j]
            else:
                pass
    if depth>1:
        out = layer_expansion
    return out


# Suppose we have an arbitrary circuit graph and the root gate is given as "root"
# We use DFS to unpack the root

# Use the example as the test case: f(x,y,z) = (xz)'(x'y')'
if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------
    # Example test case
    # gate1 = gate("None",None,None,None,"","","")
    # gate2 = gate("None",None,None,None,"","","")
    # gate3 = gate("AND",None,gate1,gate2,"(xz)*","x*y*","")
    # gate1 = gate("NAND",gate3,None,None,"x","z","(xz)*")
    # gate2 = gate("NAND",gate3,None,None,"x*","y*","x*y*")
    # gate3 = gate("AND",None,gate1,gate2,"(xz)*","x*y*","")
    #
    # circuit = [[gate3],[gate1,gate2]]

    # ------------------------------------------------------------------------------------------------
    # test case 1: f(x,y,z) = x+y
    # gate1 = gate("OR",None,None,None,"x","y","")
    # circuit = [[gate1]]

    # -----------------------------------------------------------------------------------------------
    # test case 2: f(x,y,z) = (x'y'z')'
    gate1 = gate("None",None,None,None,"","","")
    gate2 = gate("None",None,None,None,"","","")

    gate1 = gate("AND",gate2,None,None,"x*","y*","x*y*")
    gate2 = gate("NAND",None,gate1,None,"x*y*","z*","")

    circuit = [[gate2],[gate1]]
    # ------------------------------------------------------------------------------------------------
    res = unpack(circuit)
    # print(res)

    # Post processing remove padded 0s and reverse x*, y*, z*
    indices = []
    for i in range(len(res)):
        if res[i] == 0:
            indices.append(i)
    a = len(indices)-1
    while a>=0:
        index = indices[a]
        del res[index]
        a = a-1
    for i in range(len(res)):
        # print(res[i][0], res[i][0] in ['x*', 'y*', 'z*'])
        if res[i][0] in ['x*', 'y*', 'z*']:
            res[i][0] = res[i][0][0]
            tmp_perm = res[i][1][0]
            res[i][1][0] = res[i][1][1]
            res[i][1][1] = tmp_perm
        print(res[i])
