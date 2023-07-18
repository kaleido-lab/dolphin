from .inference import main
import torch

class Sadtalker:
    def __init__(self):
        self.driven_audio = './image/bus_chinese.wav'
        self.source_image = './image/art_0.png'
        self.ref_eyeblink = None
        self.ref_pose = None
        self.checkpoint_dir = './modules/sadtalker/checkpoints'
        self.result_dir = './video/sadtalker'
        self.pose_style = 0
        self.batch_size = 2
        self.size = 256
        self.expression_scale = 1.0
        self.input_yaw = None
        self.input_pitch = None
        self.input_roll = None
        self.enhancer = None
        self.background_enhancer = None
        self.preprocess = 'crop'
        self.cpu = False
        self.old_version = False
        self.still = False
        
        self.net_recon = 'resnet50'
        self.init_path = None
        self.use_last_fc = False
        self.bfm_folder = '../modules/sadtalker/checkpoints/BFM_Fitting/'
        self.bfm_model = 'BFM_model_front.mat'
        self.face3dvis = False
        self.verbose = False
        
        self.focal = 1015.0
        self.center = 112.0
        self.camera_d = 10.0
        self.z_near = 5.0
        self.z_far = 15.0

        if torch.cuda.is_available() and not self.cpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def inference(self, inputs):
        splits = inputs.split(",")
        self.driven_audio = splits[0]
        self.source_image = splits[1]
        main(self)