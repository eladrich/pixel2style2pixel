dataset_paths = {
    'celeba_train': '',
    'celeba_test': '',
    'celeba_train_sketch': '',
    'celeba_test_sketch': '',
    'celeba_train_segmentation': '',
    'celeba_test_segmentation': '',
    'ffhq': '',

    # fingerprints
    'nist_sd14_train': '/hdd/PycharmProjects/fingerprints/Data/CROPPED_NIST14/train_B',
    'nist_sd14_test': '/hdd/PycharmProjects/fingerprints/Data/CROPPED_NIST14/test_B_50',
    'nist_sd14_debug': '/hdd/PycharmProjects/fingerprints/Data/CROPPED_NIST14/test_B_2',

    # mnt
    'nist_sd14_mnt_train': '/hdd/PycharmProjects/fingerprints/Data/CROPPED_NIST14_with_ori/train_A',
    'nist_sd14_mnt_test': '/hdd/PycharmProjects/fingerprints/Data/CROPPED_NIST14_with_ori/test_A',
    'nist_sd14_mnt_gt_train': '/hdd/PycharmProjects/fingerprints/Data/CROPPED_NIST14_with_ori/train_B',
    'nist_sd14_mnt_gt_test': '/hdd/PycharmProjects/fingerprints/Data/CROPPED_NIST14_with_ori/test_B',
}

model_paths = {
    'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
    'ir_se50': '/hdd/PycharmProjects/fingerprints/pixel2style2pixel/pretrained_models/model_ir_se50.pth',
    'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
    'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
    'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
    'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
    'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
    'stylegan2_fingerprints_256': '/hdd/PycharmProjects/fingerprints/stylegan2-pytorch/checkpoint/290000.pt',
    'fingernet': '/hdd/PycharmProjects/fingerprints/pixel2style2pixel/pretrained_models/fingerNet.pth'
}
