'''
  File: utils.py
 
  Author: Thomas Kost, Mark Schelbe, Zichao Xian, Trishala Chari
  
  Date: 02 March 2022
  
  @brief acessory functions for aiding the development process
'''

import os
import platform

def add_path(file:str)->str:
    if platform.system() =='Linux':
        rel_path = "data/project/"
        return rel_path+file
    elif platform.system() == "Windows":
        rel_path = "data\project\\"
        return rel_path+file
    else:
        rel_path = "data/_MACOSX/project/"
        return rel_path+file