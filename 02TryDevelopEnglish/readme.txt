


python doccano.py \
    --doccano_file ./data/doccano_ext.json \
    --save_dir ./data \
    --splits 0.8 0.2 0 \
    --task_type ext \
    --schema_lang en


python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --batch_size 16 \
    --max_seq_len 512 \
    --multilingual

    model_zoo/uie/README.md


    
单卡启动：
```shell
python finetune.py \
    --train_path ./data/train.txt \
    --dev_path ./data/dev.txt \
    --save_dir ./checkpoint \
    --learning_rate 1e-5 \
    --batch_size 16 \
    --max_seq_len 512 \
    --num_epochs 100 \
    --model uie-base-en \
    --seed 1000 \
    --logging_steps 10 \
    --valid_steps 1000 \
    --device gpu