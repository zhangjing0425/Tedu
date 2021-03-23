import copy
import json
import time
import base64
import hmac


class Jwt:

    def __init__(self):
        pass

    @staticmethod
    def encode(self_payload,key,exp=300):

        header = {'alg':'HS256','typ':'JWT'}
        header_js = json.dumps(header,separators=(',',':')).encode()
        header_bs = Jwt.bs64encode(header_js)

        payload = copy.deepcopy(self_payload)
        payload['exp'] = time.time() + exp
        payload_js = json.dumps(payload,separators=(',',':')).encode()
        payload_bs = Jwt.bs64encode(payload_js)

        hm_js = hmac.new(key.encode(),header_bs + b'.' + payload_bs,digestmod='SHA256')
        hm_bs = Jwt.bs64encode(hm_js.digest())

        return header_bs + b'.' + payload_bs + b'.' + hm_bs

        # header = {'alg':'HS256','typ':'JWT'}
        # header_js = json.dumps(header,separators=(',',':'),sort_keys=True)
        # header_bs = Jwt.bs64encode(header_js.encode())
        #
        # payload = copy.deepcopy(self_payload)
        # payload['exp'] = exp + time.time()
        # payload_js = json.dumps(payload,separators=(',',':'),sort_keys=True)
        # payload_bs = Jwt.bs64encode(payload_js.encode())
        #
        # hm = hmac.new(key.encode(),header_bs + b'.' + payload_bs,digestmod='SHA256')
        # hm_bs = Jwt.bs64encode(hm.digest())
        #
        # return header_bs + b'.' + payload_bs + b'.' + hm_bs

    @staticmethod
    def bs64encode(js):

        return base64.urlsafe_b64encode(js).replace(b'=',b'')

    @staticmethod
    def bs64decode(bs):

        rem = len(bs) % 4
        if rem > 0:
            bs += b'=' * (4-rem)
        return base64.urlsafe_b64decode(bs)

    @staticmethod
    def decode(token,key):
        header_bs,payload_bs,sign_bs = token.split(b'.')
        hm_bs = hmac.new(key.encode(),header_bs + b'.' + payload_bs,digestmod='SHA256')
        if sign_bs != Jwt.bs64encode(hm_bs.digest()):
            raise
        payload_js = Jwt.bs64decode(payload_bs)
        payload = json.loads(payload_js)
        exp = payload['exp']
        now = time.time()
        if now > exp:
            raise
        return payload
s = Jwt.encode({'name':'zhangjing'},'123456')
print(s)

print('------------')
print(Jwt.decode(s,'123456'))

