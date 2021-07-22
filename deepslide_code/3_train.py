#import config_debug as config
import config
from utils_model_STN import train_resnet#
import os
from pathlib import Path
from utils import get_log_csv_name
# Training the ResNet.
'''
print("\n\n+++++ Running 3_train.py +++++")
train_resnet(batch_size=config.args.batch_size,
             checkpoints_folder=config.args.checkpoints_folder,
             classes=config.classes,
             color_jitter_brightness=config.args.color_jitter_brightness,
             color_jitter_contrast=config.args.color_jitter_contrast,
             color_jitter_hue=config.args.color_jitter_hue,
             color_jitter_saturation=config.args.color_jitter_saturation,
             device=config.device,
             learning_rate=config.args.learning_rate,
             learning_rate_decay=config.args.learning_rate_decay,
             log_csv=config.log_csv,
             num_classes=config.num_classes,
             num_layers=config.args.num_layers,
             num_workers=config.args.num_workers,
             path_mean=config.path_mean,
             path_std=config.path_std,
             pretrain=config.args.pretrain,
             resume_checkpoint=config.args.resume_checkpoint,
             resume_checkpoint_path=config.resume_checkpoint_path,
             save_interval=config.args.save_interval,
             num_epochs=config.args.num_epochs,
             train_folder=config.args.train_folder,
             weight_decay=config.args.weight_decay)
print("+++++ Finished running 3_train.py +++++\n\n")
'''

os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
print("\n\n+++++ Running 3_train.py +++++")
tilespath='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/512px_Tiled_preweight_deepslide'
#'/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/512px_Tiled_train_valid_bagging_patient_level_20Xfro5X_deepslide'
#'/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/512px_Tiled_train_valid_bagging_patient_level_deepslide'
config.args.train_folder=Path(tilespath)
log_csv='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/bagging_deepslide_preweight_STN'
log_folder=Path(log_csv)
config.args.checkpoints_folder=Path(os.path.join(log_csv,'checkpoints'))
config.args.log_csv = get_log_csv_name(log_folder=log_folder)
train_resnet(batch_size=config.args.batch_size,
             checkpoints_folder=config.args.checkpoints_folder,
             classes=config.classes,
             color_jitter_brightness=config.args.color_jitter_brightness,
             color_jitter_contrast=config.args.color_jitter_contrast,
             color_jitter_hue=config.args.color_jitter_hue,
             color_jitter_saturation=config.args.color_jitter_saturation,
             device=config.device,
             learning_rate=config.args.learning_rate,
             learning_rate_decay=config.args.learning_rate_decay,
             log_csv=config.args.log_csv,
             num_classes=config.num_classes,
             num_layers=config.args.num_layers,
             num_workers=config.args.num_workers,
             path_mean=config.path_mean,
             path_std=config.path_std,
             pretrain=config.args.pretrain,
             resume_checkpoint=config.args.resume_checkpoint,
             resume_checkpoint_path=config.resume_checkpoint_path,
             save_interval=config.args.save_interval,
             num_epochs=config.args.num_epochs,
             train_folder=config.args.train_folder,
             weight_decay=config.args.weight_decay)
print("+++++ Finished running 3_train.py +++++\n\n")

'''
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
print("\n\n+++++ Running bagging 3_train.py +++++")
for i in range(10):
    if i<=5:
        continue
    tilespath='/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/512px_Tiled_train_valid_bagging_patient_level_deepslide'
    #'/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/512px_Tiled_train_valid_bagging_patient_level_20Xfro5X_deepslide'
    #'/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/512px_Tiled_train_valid_bagging_patient_level_deepslide'
    config.args.train_folder=Path(os.path.join(tilespath,str(i)))
    log_csv='/disk2/zhangyingxin/xinjiang_paper_supply/5X/deepslide/STN/'
    #'/disk1/zhangyingxin/project/lung/xinjiang_grade2/second_instalment/train_valid_test/bagging_deepslide_patient_level_preweight_model21'
    log_folder=Path(os.path.join(log_csv,str(i)))
    config.args.checkpoints_folder=Path(os.path.join(log_csv,str(i),'checkpoints'))
    config.args.log_csv = get_log_csv_name(log_folder=log_folder)
    train_resnet(batch_size=config.args.batch_size,
                checkpoints_folder=config.args.checkpoints_folder,
                classes=config.classes,
                color_jitter_brightness=config.args.color_jitter_brightness,
                color_jitter_contrast=config.args.color_jitter_contrast,
                color_jitter_hue=config.args.color_jitter_hue,
                color_jitter_saturation=config.args.color_jitter_saturation,
                device=config.device,
                learning_rate=config.args.learning_rate,
                learning_rate_decay=config.args.learning_rate_decay,
                log_csv=config.args.log_csv,
                num_classes=config.num_classes,
                num_layers=config.args.num_layers,
                num_workers=config.args.num_workers,
                path_mean=config.path_mean,
                path_std=config.path_std,
                pretrain=config.args.pretrain,
                resume_checkpoint=config.args.resume_checkpoint,
                resume_checkpoint_path=config.resume_checkpoint_path,
                save_interval=config.args.save_interval,
                num_epochs=config.args.num_epochs,
                train_folder=config.args.train_folder,
                weight_decay=config.args.weight_decay)
    print("Finished ",str(i),'exam ##########################')
print("+++++ Finished running bagging 3_train.py +++++\n\n")
'''