import os
import gzip
import json
import numpy as np

DATA_DIR = 'data'
LANGUAGE = 'python' 


def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


with open(os.path.join(DATA_DIR, 'train_valid', LANGUAGE, 'train.txt'), 'r', encoding='utf-8') as f:
    train_data = f.read()
    train_data = train_data.split('\n')
    train_data = [t.split('<CODESPLIT>') for t in train_data if t != '']
    train_data = [t for t in train_data if len(t) == 5]
    train_data = [t for t in train_data if t[0] == '1']


with open(os.path.join(DATA_DIR, 'train_valid', LANGUAGE, 'valid.txt'), 'r', encoding='utf-8') as f:
    valid_data = f.read()
    valid_data = valid_data.split('\n')
    valid_data = [v.split('<CODESPLIT>') for v in valid_data if v != '']
    valid_data = [v for v in valid_data if len(v) == 5]
    valid_data = [v for v in valid_data if v[0] == '1']


with gzip.open(os.path.join(DATA_DIR, f'{LANGUAGE}_test_0.jsonl.gz'), 'r') as pf, open(os.path.join(DATA_DIR, 'test', LANGUAGE, 'test.txt'), 'a', encoding='utf-8') as f:
    data = pf.readlines()
    data = np.array(data, dtype=object)
    test_data = []
    for d in data:
        # label, url, func_name, doc, code
        d_line = json.loads(str(d, encoding='utf-8'))
        _url = d_line['url']
        _func_name = d_line['func_name']
        _docstring = ' '.join(d_line['docstring_tokens'])
        _code = ' '.join([format_str(token) for token in d_line['code_tokens']])
        _label = '1'
        test_data.append([_label, _url, _func_name, _docstring, _code])

        f.write(f"{_label}<CODESPLIT>{_url}<CODESPLIT>{_func_name}<CODESPLIT>{_docstring}<CODESPLIT>{_code}\n")



print(f"Train: {len(train_data)}")
print(f"Valid: {len(valid_data)}")
print(f"Test: {len(test_data)}")

print(train_data[0])
print(valid_data[0])
print(test_data[0])