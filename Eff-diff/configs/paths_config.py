DATASET_PATHS = {
	'FFHQ': 'data/celeba_hq/',
	'CelebA_HQ': '../../sheldon_sda1/diffusion_data/celeba_hq/',
	'AFHQ': '../../sheldon_sda1/diffusion_data/afhq',
	'LSUN':  '../../sheldon_sda1/diffusion_data/lsun',
    'IMAGENET': '/mnt/data/imagenet',
}

MODEL_PATHS = {
	'AFHQ': "pretrained/afhq_dog_4m.pt",
	'FFHQ': "pretrained/ffhq_10m.pt",
	'ir_se50': 'pretrained/model_ir_se50.pth',
    'IMAGENET': "pretrained/512x512_diffusion.pt",
	'shape_predictor': "pretrained/shape_predictor_68_face_landmarks.dat.bz2",
}


HYBRID_MODEL_PATHS = [
	'./checkpoint/human_face/curly_hair_t401.pth',
	'./checkpoint/human_face/with_makeup_t401.pth',
]

HYBRID_CONFIG = \
	{ 300: [0.4, 0.6, 0],
	    0: [0.15, 0.15, 0.7]}
