#지능형로봇 YOLOMT 구현 

conda activate robotenv

(robotenv) device08@ubuntu:~/ir-guide$ bash ./jetson_system_bridge.sh

(robotenv) device08@ubuntu:~/ir-guide$ python -c "import tensorrt as trt; print('TRT OK', trt.__version__)"
(robotenv) device08@ubuntu:~/ir-guide$ python -c "import pycuda.autoinit, pycuda.driver as cuda; print('PYCUDA OK, devices =', cuda.Device.count())"
로 버전 확인

python yolomt_inference.py --engine yolomt_best.engine
