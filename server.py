from waitress import serve
import evaprojector

serve(evaprojector.app, host='0.0.0.0', port=8080)


#TO RUN IN PYCHARM:
#Command prompt 01 Hit play server.py in pycharm
#C:\Anaconda3\envs\styleGAN2\python.exe D:/Work/SCIArc/2022Summer_Web/pyscript/server.py
#Command prompt 02
#(styleGAN2) D:\Work\SCIArc\2022Summer_Web\pyscript>python -m http.server