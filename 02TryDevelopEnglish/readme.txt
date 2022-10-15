

1.数据预处理
python doccano.py \
    --doccano_file ./data/doccano_ext25.json \
    --save_dir ./data \
    --splits 0.8 0.2 0 \
    --task_type ext \
    --schema_lang en


2.模型微调之单卡启动：

nohup python finetune.py \
    --train_path ./data/train.txt \
    --dev_path ./data/dev.txt \
    --save_dir ./checkpoint \
    --learning_rate 1e-5 \
    --batch_size 2 \
    --max_seq_len 2048 \
    --num_epochs 100 \
    --model uie-base-en \
    --seed 1000 \
    --logging_steps 10 \
    --valid_steps 100 \
    --device gpu > train_2022_10_15_03train.log &


    2048?512? 改batch_size从16为8 试试1024 2048(2)



3.模型评估
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --batch_size 16 \
    --max_seq_len 512 > eva_2022_10_15.log &


测试zero-shot则用uie-base-en
    model_zoo/uie/README.md


    


