import os

path_to_examples_repository = r'../../podpac_examples'

base_link = 'https://github.com/creare-com/podpac-examples/blob/develop/notebooks'

nbpath = os.path.join(path_to_examples_repository, 'notebooks')
files = os.listdir(nbpath)

prestring = '* '
string = '\n'.join([prestring + ' [{}]({})'.format(f, base_link + '/' + f) for f in files if f.endswith('ipynb')])

subdirs = [f for f in files if os.path.isdir(os.path.join(nbpath, f)) and f not in ['Images', 'old_examples', 'presentations']]

for sd in subdirs:
    path = os.path.join(nbpath, sd)
    link = base_link + '/' + sd
    fs = os.listdir(path)
    string += '\n* `{}`\n'.format(sd)
    prestring = '   *'
    string += '\n'.join([prestring + ' [{}]({})'.format(f, link + '/' + f) for f in fs if f.endswith('ipynb')])
    
print(string)