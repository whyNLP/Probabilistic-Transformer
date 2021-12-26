from pathlib import Path
from typing import List
import toml, os

def readconfig(config_path):
    if not os.path.exists(config_path):
        raise Exception('config file not exists')
    with open(config_path, mode='rb') as f:
        content = f.read()
    if content.startswith(b'\xef\xbb\xbf'):     # 去掉 utf8 bom 头
        content = content[3:]
    dic = toml.loads(content.decode('utf8'))
    return dic

def saveconfig(config, save_file):
    """
    config is a dictionary.
    save_path: saving path include file name.
    """
    if isinstance(save_file, str):
        save_file = Path(save_file)
    if not save_file.parent.exists():
        os.makedirs(save_file.parent)
    if isinstance(config, dict):
        with open(save_file, 'w') as f:
            toml.dump(config, f)
    else:
        os.system(" ".join(["cp", str(config), str(save_file)]))

def set_cuda_id(cuda_id):
    if isinstance(cuda_id, int):
        cuda_id = str(cuda_id)
    os.environ['CUDA_VISIBLE_DEVICES']=cuda_id

def getattrs(__os: List[object], name: str):
    for __o in __os:
        obj = getattr(__o, name, None)
        if obj is not None:
            return obj
    raise ModuleNotFoundError(f"All objects do not have '{name}'")
