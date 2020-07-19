# train
python3 unet.py --do_train \
                --mask_json_path 'mask.json' \
                --batch_size 128 \
                --epochs 20 \
                --save_epoch 2 \
                --do_save_summary \
                --summary_image './summary.png'
# test
python3 unet.py --do_test \
                --mask_json_path 'mask.json' \
                --model_state_dict ./model/checkpoint_2.pth \
                --test_image ./dataset/images/00000001.jpg \
                --test_save_path ./test.jpg

# Local Train
# python unet.py --do_train --batch_size 1

# Using ./dataset/gt/00002338.png as mask colors
# Mask colors: [(255.0, 255.0, 255.0), (0.0, 0.0, 255.0), (255.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 255.0, 0.0)]
# Train dataset: image count = 50
# Test dataset: image count = 20
# Mask colors list has been saved in ./mask.json
# Start training...

# Local Test
# python unet.py --do_test --model_state_dict ./model/checkpoint_2.pth --test_image ./dataset/images/00000001.jpg --test_save_path ./test.jpg

# Loading model param from ./model/checkpoint_2.pth
# Reading mask colors list from ./mask.json
# Mask colors: [(255.0, 255.0, 255.0), (0.0, 0.0, 255.0), (255.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 255.0, 0.0)]