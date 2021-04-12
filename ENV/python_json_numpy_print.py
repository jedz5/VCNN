import json, sys
import numpy as np
import pprint as pp

# Set numpy's printoptions to display all the data with max precision
np.set_printoptions(threshold=np.inf,
                    linewidth=sys.maxsize,
                    suppress=True,
                    nanstr='0.0',
                    infstr='0.0',
                    precision=np.finfo(np.longdouble).precision)



# Modified version of Adam Hughes's https://stackoverflow.com/a/27948073/1429402
def save_formatted(fname,data):

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return '__ndarray__'+self.numpy_to_string(obj)

            return json.JSONEncoder.default(self, obj)


        def numpy_to_string(self,data):
            ''' Use pprint to generate a nicely formatted string
            '''

            # Get rid of array(...) and keep only [[...]]
            f = pp.pformat(data, width=sys.maxsize)
            f = f[6:-1].splitlines() # get rid of array(...) and keep only [[...]]

            # Remove identation caused by printing "array("
            for i in range(1,len(f)):
                f[i] = f[i][6:]

            return '\n'.join(f)


    # Parse json stream and fix formatting.
    # JSON doesn't support float arrays written as [0., 0., 0.]
    # so we look for the problematic numpy print syntax and correct
    # it to be readable natively by JSON, in this case: [0.0, 0.0, 0.0]
    with open(fname,'w') as io:
        for line in json.dumps(data, sort_keys=False, indent=4, cls=NumpyEncoder).splitlines():
            if '__ndarray__' in line:
                index = line.index('"__ndarray__')
                lines = line.replace('"__ndarray__','') #remove '__ndarray__' at the end of the last array
                lines = lines.replace(']"',']') #remove '"' at the end of the last array
                lines = lines.replace('. ','.0')  # convert occurences of ". " to ".0"    ex: 3. , 2. ]
                lines = lines.replace('.,','.0,') # convert occurences of ".," to ".0,"   ex: 3., 2.,
                lines = lines.replace('.]','.0]') # convert occurences of ".]" to ".0],"  ex: 3., 2.]
                lines = lines.split('\\n')


                # write each lines with appropriate indentation
                for i in range(len(lines)):
                    if i == 0:
                        io.write(lines[i]+"\n")
                    else:
                        indent = ' '*(index)
                        io.write('%s%s\n'%(indent,lines[i]))

            else:
                io.write('%s\n'%line)



def load_formatted(fname):

    def json_numpy_obj_hook(dct):
        if isinstance(dct, dict) and '__ndarray__' in dct:
            return np.array(dct['__ndarray__']).astype(dct['__dtype__'])
        return dct
if __name__ == '__main__':
    jdata = {"round": 0,
                "stacks": np.array([[1, 1, 40, 40, 3, 3, 2, 2, 3, 1, 0, 0, 0, 0, 9, 0, 0, 16, 8, 15, 119],
                [0, 0, 14, 14, 10, 10, 6, 3, 3, 2, 0, 0, 0, 0, 6, 0, 0, 16, 0, 1, 3],
                [0, 1, 1, 1, 10, 10, 6, 5, 3, 2, 0, 0, 0, 0, 5, 0, 0, 16, 2, 1, 1],
                [0, 2, 1, 1, 10, 10, 6, 5, 3, 2, 0, 0, 0, 0, 5, 0, 0, 16, 4, 1, 1],
                [0, 3, 47, 47, 10, 10, 6, 5, 3, 2, 0, 0, 0, 0, 5, 0, 0, 16, 5, 1, 1],
                [0, 4, 1, 1, 10, 10, 6, 5, 3, 2, 0, 0, 0, 0, 5, 0, 0, 16, 6, 1, 1],
                [0, 5, 1, 1, 10, 10, 6, 5, 3, 2, 0, 0, 0, 0, 5, 0, 0, 16, 8, 1, 1],
                [0, 6, 1, 1, 10, 10, 6, 5, 3, 2, 0, 0, 0, 0, 5, 0, 0, 16, 10, 1, 1],
                [1, 0, 40, 40, 3, 3, 2, 2, 3, 1, 0, 0, 0, 1, 9, 0, 0, 16, 2, 15, 119],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
    save_formatted("d:/test.json",jdata)