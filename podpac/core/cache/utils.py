
import hashlib
import json

def is_json_serializable(obj):
    try:
        json.dumps(obj)
    except:
        return False
    else:
        return True

def hash_string(s):
    return hashlib.md5(s.encode()).hexdigest()