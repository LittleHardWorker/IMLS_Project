class Config():
    def __init__(self, task):
        self.task = task
        if self.task == 'pretrain':
            self.cifar10 = False
            self.model = '18'
            self.num_classes = 5
            self.train_file_path = ''
            self.learning_rate=0.1
            self.blur = True
            self.noise = True
            self.flip = True
            self.crop = True
            self.rotate = False
            self.jitter = True
            self.colordrop = True
            self.cutout = False
            self.batch_size = 64   
            self.num_epochs = 1000
            self.rotate_prob = 0.5
            self.max_rotation = 30
            self.flip_prob = 0.5
            self.blur_prob = 0.5
            self.color_prob = 0.2
            self.jitter_prob = 0.8
            self.cutout_prob = 0.5
            self.strength  = 1.0
            self.pretrain_save_path = 'logs/pretrain'
            self.input_size = [500, 500]
            self.crop_size = [240, 240]
            self.pretrained = False
            self.freeze = False
            self.include_nonlinearity=False
            self.temperature = 0.5
            self.weight_decay=1e-6
            self.zdim = 128

            self.random_HSV_in_YIQ = False
            self.random_HSV_in_YIQ_prob = 0.5
            self.dense_image_warp = False
            self.dense_image_warp_prob = 0.5
            self.euc_dist_trans = False
            self.euc_dist_trans_prob = 0.5
            self.sharpness = False
            self.sharpness_prob = 0.5

            self.proj_head_act = 'relu'




        elif self.task == "segmentation" or self.task == "classification":
            self.cifar10 = False
            self.model = '18'
            self.num_classes = 5
            self.imagenet_path = ''
            self.train_file_path = ''
            self.val_file_path = ''
            self.learning_rate=0.001
            self.blur = False
            self.noise = False
            self.flip = False
            self.crop = False
            self.rotate = False
            self.jitter = False
            self.colordrop = False
            self.cutout = False
            self.batch_size = 64   
            self.num_epochs = 100
            self.rotate_prob = 0.5
            self.max_rotation = 30
            self.flip_prob = 0.5
            self.blur_prob = 0.5
            self.color_prob = 0.2
            self.jitter_prob = 0.8
            self.cutout_prob = 0.5
            self.strength  = 1.0
            self.pretrain_save_path = 'logs/pretrain'
            self.finetune_save_path = 'logs/finetune'
            self.input_size = [500, 500]
            self.crop_size = [240, 240]
            self.pretrained = False
            self.freeze = False
            self.include_nonlinearity=False
            self.weight_decay=0
            self.zdim = 128

            self.random_HSV_in_YIQ = False
            self.random_HSV_in_YIQ_prob = 0.5
            self.dense_image_warp = False
            self.dense_image_warp_prob = 0.5
            self.euc_dist_trans = False
            self.euc_dist_trans_prob = 0.5
            self.sharpness = False
            self.sharpness_prob = 0.5

            self.proj_head_act = 'relu'



        else:
            raise ValueError("Task not supported, must be either pretrain, classification or segmentation")
