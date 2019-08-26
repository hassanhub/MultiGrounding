from groundnet import GroundNet
from trainers import gnet_trainer, gnet_trainer_distributed

path_dict = {'lmdb': '../data/saved/vg/images.lmdb',
             'annotations': '../data/saved/vg/annotations/train.pickle'}

trainer = gnet_trainer_distributed(data_path = path_dict,
                                   gpu='2,3',
                                   batch_size=32,
                                   lr=0.0001,
                                   decay_steps=10000,
                                   decay_rate=1,
                                   grdnt_clip=False,
                                   grdnt_rng=10.,
                                   gnet_config='./configs/pnas_elmo_1x1.yml',
                                   log_step=2000)

trainer.train(epochs=10,ckpt_path='./models/groundnet_pnas_elmo_1x1_vg')