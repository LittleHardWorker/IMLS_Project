# 1. Running Environment
###This project is designed in python 3.6. 
###Debug through PyCharm Community Edition 2020.2.3 x64


#2. Usage of Code
###The following code can be used to reproduce the results in the report
## Benchmark
###Pretrain:
###python main.py --task=pretrain --cifar10=True --batch_size=256 --num_epochs=100 --temperature=0.1 --strength=0.5 --learning_rate=1.0
###Finetune:
###python main.py --task=classification --cifar10=True --batch_size=256 --num_classes=10 --pretrain_save_path=logs/pretrain --finetune_save_path=logs/finetune --freeze 
## Task A
###Add input parameter ¡°¡ªnew_aug = x¡± at the end of both parts. 
###For Random HSV in YIQ, x=1
###For Dense Image Warp, x=2
###For Sharpness, x=4
## Task B
###Add input parameter ¡°¡ªmodel = y¡± at the end of both parts. 
###For ResNet-18, y=18 (default value of ¡°model¡± is 18)
###For ResNet-34, y=34
###
## Task C
###Add input parameter ¡°¡ªmodel = z¡± at the end of both parts. 
###For ResNet-50, z=50
###For ResNeXt-50, z=ResNeXt_50
## Task D
###Add input parameter ¡°¡ªproj_head_act= a¡± at the end of both parts. 
###For ReLU, a=relu (default value of ¡°proj_head_act¡± is relu)
###For tanh, a=tanh
###For sigmoid, a= sigmoid
###For softmax, a= softmax




