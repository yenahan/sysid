python -u test_artificial.py --config-name RidgeModelArgument --metric cv_ridge --img_set imagenet --num_imgs 3000 --srp_dim 6862 --source_model vgg11 --target_model alexnet --gpu_index 1 --savefolder ridge_cv/imagenet > artificial_score/output/cv_ridge/vgg11_alex_cvridge.log 2>&1;
python -u test_artificial.py --config-name RidgeModelArgument --metric cv_ridge --img_set imagenet --num_imgs 3000 --srp_dim 6862 --source_model resnet18 --target_model alexnet --gpu_index 1 --savefolder ridge_cv/imagenet > artificial_score/output/cv_ridge/resnet18_alex_cvridge.log 2>&1;
python -u test_artificial.py --config-name RidgeModelArgument --metric cv_ridge --img_set imagenet --num_imgs 3000 --srp_dim 6862 --source_model vit_b_32 --target_model alexnet --gpu_index 1 --savefolder ridge_cv/imagenet > artificial_score/output/cv_ridge/vit_b_32_alex_cvridge.log 2>&1;
python -u test_artificial.py --config-name RidgeModelArgument --metric cv_ridge --img_set imagenet --num_imgs 3000 --srp_dim 6862 --source_model mixer_b16_224 --target_model alexnet --source_pckg timm --gpu_index 1 --savefolder ridge_cv/imagenet > artificial_score/output/cv_ridge/mixer_alex_cvridge.log 2>&1;
