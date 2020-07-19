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
                --test_image ./test.jpg \
                --test_save_path ./test_seg.jpg
