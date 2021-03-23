from socket import *

socketfd = socket(AF_INET,SOCK_STREAM)

socketfd.connect(('127.0.0.1',8888))

msg = input('>>')

socketfd.send(msg.encode())

data = socketfd.recv(1024).decode()

print('data',data)

socketfd.close()
