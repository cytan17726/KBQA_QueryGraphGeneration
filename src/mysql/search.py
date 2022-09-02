import pymysql
import sys
import os
import json
from typing import List, Dict, Tuple, Optional
import copy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)