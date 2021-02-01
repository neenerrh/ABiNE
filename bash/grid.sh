#! /bin/bash
#source activate new_BiNE
#mkdir bash/log
#python model/train.py --alpha --beta --gamma --lam --lr --ws --ns --walk_length --number_walks --max_iter --d --top_N --ABRW_topk
echo '......ABiNERec  bash......'

echo '------------ Abinerec result-----------------------------------------'
echo '------------ Abinerec result-----------------------------------------' > bash/log/ABiNEgrid.txt
start=`date +%s`
python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --u_attribute_file data/mooc/months1/u_attr.txt --v_attribute_file data/mooc/months1/v_attr.txt   --alpha 0.01 --beta 0.01 --gamma 0.1 --lam 0.025 --lr 0.01 --ws 5 --ns 4 --walk-length 80 --number-walks 10 --max-iter 50 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNEgrid.txt
#python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --lam 0.05 --max-iter 50 --ns 4 --ws 5 --lr 0.01 --model-name data/mooc --rec 1 --large 0 --vectors-u data/mooc/months1/vectors_u.dat --vectors-v data/mooc/months1/vectors_v.dat >> bash/log/ABiNERec_1.txt
end=`date +%s`
echo ALL running time: $((end-start)) >> bash/log/AABiNEgrid.txt
python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --u_attribute_file data/mooc/months1/u_attr.txt --v_attribute_file data/mooc/months1/v_attr.txt --alpha 0.01 --beta 0.01 --gamma 0.5 --lam 0.025 --lr 0.01  --ws 5 --ns 4 --walk-length 80 --number-walks 10 --max-iter 50 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNEgrid.txt
#python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --lam 0.025 --max-iter 50 --model-name data/mooc --rec 1 --large 0 --vectors-u data/mooc/months1/vectors_u.dat --vectors-v data/mooc/months1/vectors_v.dat >> bash/log/ABiNERec_1.txt
echo '--done--' >> bash/log/ABiNEgrid.txt
start=`date +%s`
python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --u_attribute_file data/mooc/months1/u_attr.txt --v_attribute_file data/mooc/months1/v_attr.txt --alpha 0.01 --beta 0.01 --gamma 0.1 --lam 0.01 --lr 0.01 --ws 5 --ns 4 --walk-length 80 --number-walks 10 --max-iter 50 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNEgrid.txt
#python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --lam 0.05 --max-iter 50 --ns 4 --ws 5 --lr 0.01 --model-name data/mooc --rec 1 --large 0 --vectors-u data/mooc/months1/vectors_u.dat --vectors-v data/mooc/months1/vectors_v.dat >> bash/log/ABiNERec_1.txt
end=`date +%s`
echo ALL running time: $((end-start)) >> bash/log/ABiNEgrid.txt
python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --u_attribute_file data/mooc/months1/u_attr.txt --v_attribute_file data/mooc/months1/v_attr.txt --alpha 0.01 --beta 0.01 --gamma 0.5 --lam 0.01 --lr 0.01  --ws 5 --ns 4 --walk-length 80 --number-walks 10 --max-iter 50 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNEgrid.txt
#python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --lam 0.025 --max-iter 50 --model-name data/mooc --rec 1 --large 0 --vectors-u data/mooc/months1/vectors_u.dat --vectors-v data/mooc/months1/vectors_v.dat >> bash/log/ABiNERec_1.txt
echo '--done--' >> bash/log/ABiNEgrid.txt
start=`date +%s`
python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --u_attribute_file data/mooc/months1/u_attr.txt --v_attribute_file data/mooc/months1/v_attr.txt --alpha 0.01 --beta 0.01 --gamma 0.1 --lam 0.05 --lr 0.01 --ws 5 --ns 4 --walk-length 80 --number-walks 10 --max-iter 50 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNEgrid.txt
#python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --lam 0.05 --max-iter 50 --ns 4 --ws 5 --lr 0.01 --model-name data/mooc --rec 1 --large 0 --vectors-u data/mooc/months1/vectors_u.dat --vectors-v data/mooc/months1/vectors_v.dat >> bash/log/ABiNERec_1.txt
end=`date +%s`
echo ALL running time: $((end-start)) >> bash/log/ABiNEgrid.txt
python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --u_attribute_file data/mooc/months1/u_attr.txt --v_attribute_file data/mooc/months1/v_attr.txt --alpha 0.01 --beta 0.01 --gamma 0.5 --lam 0.05 --lr 0.01  --ws 5 --ns 4 --walk-length 80 --number-walks 10 --max-iter 50 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNEgrid.txt
#python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --lam 0.025 --max-iter 50 --model-name data/mooc --rec 1 --large 0 --vectors-u data/mooc/months1/vectors_u.dat --vectors-v data/mooc/months1/vectors_v.dat >> bash/log/ABiNERec_1.txt
echo '--done--' >> bash/log/ABiNEgrid.txt
start=`date +%s`
python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --u_attribute_file data/mooc/months1/u_attr.txt --v_attribute_file data/mooc/months1/v_attr.txt --alpha 0.01 --beta 0.01 --gamma 0.1 --lam 0.025 --lr 0.025 --ws 5 --ns 4 --walk-length 80 --number-walks 10 --max-iter 50 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNEgrid.txt
#python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --lam 0.05 --max-iter 50 --ns 4 --ws 5 --lr 0.01 --model-name data/mooc --rec 1 --large 0 --vectors-u data/mooc/months1/vectors_u.dat --vectors-v data/mooc/months1/vectors_v.dat >> bash/log/ABiNERec_1.txt
end=`date +%s`
echo ALL running time: $((end-start)) >> bash/log/ABiNEgrid.txt
python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --u_attribute_file data/mooc/months1/u_attr.txt --v_attribute_file data/mooc/months1/v_attr.txt --alpha 0.01 --beta 0.01 --gamma 0.5 --lam 0.025 --lr 0.05  --ws 5 --ns 4 --walk-length 80 --number-walks 10 --max-iter 50 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNEgrid.txt
#python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --lam 0.025 --max-iter 50 --model-name data/mooc --rec 1 --large 0 --vectors-u data/mooc/months1/vectors_u.dat --vectors-v data/mooc/months1/vectors_v.dat >> bash/log/ABiNERec_1.txt
echo '--done--' >> bash/log/ABiNEgrid.txt
start=`date +%s`
python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --u_attribute_file data/mooc/months1/u_attr.txt --v_attribute_file data/mooc/months1/v_attr.txt --alpha 0.01 --beta 0.01 --gamma 0.1 --lam 0.025 --lr 0.01 --ws 5 --ns 4 --walk-length 80 --number-walks 5 --max-iter 50 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNEgrid.txt
#python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --lam 0.05 --max-iter 50 --ns 4 --ws 5 --lr 0.01 --model-name data/mooc --rec 1 --large 0 --vectors-u data/mooc/months1/vectors_u.dat --vectors-v data/mooc/months1/vectors_v.dat >> bash/log/ABiNERec_1.txt
end=`date +%s`
echo ALL running time: $((end-start)) >> bash/log/ABiNEgrid.txt
python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --u_attribute_file data/mooc/months1/u_attr.txt --v_attribute_file data/mooc/months1/v_attr.txt --alpha 0.01 --beta 0.01 --gamma 0.5 --lam 0.025 --lr 0.01  --ws 5 --ns 4 --walk-length 80 --number-walks 3 --max-iter 50 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNEgrid.txt
#python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --lam 0.025 --max-iter 50 --model-name data/mooc --rec 1 --large 0 --vectors-u data/mooc/months1/vectors_u.dat --vectors-v data/mooc/months1/vectors_v.dat >> bash/log/ABiNERec_1.txt
echo '--done--' >> bash/log/ABiNEgrid.txt
start=`date +%s`
python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --u_attribute_file data/mooc/months1/u_attr.txt --v_attribute_file data/mooc/months1/v_attr.txt --alpha 0.01 --beta 0.01 --gamma 0.1 --lam 0.025 --lr 0.01 --ws 5 --ns 4 --walk-length 80 --number-walks 5 --max-iter 100 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNEgrid.txt
#python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --lam 0.05 --max-iter 50 --ns 4 --ws 5 --lr 0.01 --model-name data/mooc --rec 1 --large 0 --vectors-u data/mooc/months1/vectors_u.dat --vectors-v data/mooc/months1/vectors_v.dat >> bash/log/ABiNERec_1.txt
end=`date +%s`
echo ALL running time: $((end-start)) >> bash/log/ABiNEgrid.txt
python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --u_attribute_file data/mooc/months1/u_attr.txt --v_attribute_file data/mooc/months1/v_attr.txt --alpha 0.01 --beta 0.01 --gamma 0.5 --lam 0.025 --lr 0.01  --ws 5 --ns 4 --walk-length 80 --number-walks 3 --max-iter 100 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNEgrid.txt
#python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --lam 0.025 --max-iter 50 --model-name data/mooc --rec 1 --large 0 --vectors-u data/mooc/months1/vectors_u.dat --vectors-v data/mooc/months1/vectors_v.dat >> bash/log/ABiNERec_1.txt
echo '--done--' >> bash/log/ABiNEgrid.txt
start=`date +%s`
python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --u_attribute_file data/mooc/months1/u_attr.txt --v_attribute_file data/mooc/months1/v_attr.txt --alpha 0.01 --beta 0.01 --gamma 0.1 --lam 0.025 --lr 0.01 --ws 5 --ns 4 --walk-length 80 --number-walks 5 --max-iter 100 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.4 >> bash/log/ABiNEgrid.txt
#python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --lam 0.05 --max-iter 50 --ns 4 --ws 5 --lr 0.01 --model-name data/mooc --rec 1 --large 0 --vectors-u data/mooc/months1/vectors_u.dat --vectors-v data/mooc/months1/vectors_v.dat >> bash/log/ABiNERec_1.txt
end=`date +%s`
echo ALL running time: $((end-start)) >> bash/log/ABiNEgrid.txt
python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --u_attribute_file data/mooc/months1/u_attr.txt --v_attribute_file data/mooc/months1/v_attr.txt --alpha 0.01 --beta 0.01 --gamma 0.5 --lam 0.025 --lr 0.01  --ws 5 --ns 4 --walk-length 80 --number-walks 3 --max-iter 100 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.5 >> bash/log/ABiNEgrid.txt
#python model/train.py --train-data data/mooc/months1/ratings_train.csv --test-data data/mooc/months1/ratings_test.csv --lam 0.025 --max-iter 50 --model-name data/mooc --rec 1 --large 0 --vectors-u data/mooc/months1/vectors_u.dat --vectors-v data/mooc/months1/vectors_v.dat >> bash/log/ABiNERec_1.txt
echo '--done--' >> bash/log/ABiNEgrid.txt
echo '......well done......'



