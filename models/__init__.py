# Python 包的标准做法，使外部代码可以通过 from models import build_model 来使用模型。
from .decoder import build
from .Edffn import EDFFN  # 确保 EDFFN 模块可以被正确导入

def build_model(args):
    return build(args)

