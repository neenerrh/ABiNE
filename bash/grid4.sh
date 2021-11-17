#! /bin/bash
#source activate new_BiNE
#mkdir bash/log

echo '......ABiNENegbatch bash......'

echo '------------ ABiNENegbatch.txt-----------------------------------------'
echo '------------ABiNENegbatch.txt----------------------------------------' > bash/log/ABiNENegbatch.txt
start=`date +%s`
python model/train.py --train-data data/mooc/months1/sample1/ratings_train.csv --test-data data/mooc/months1/sample1/ratings_test.csv --alpha 0.01 --beta 0.01 --gamma 0.2 --lam 0.01 --lr 0.01 --ws 5 --ns 4 --walk-length 80 --number-walks 3 --max-iter 200 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNENegbatch.txt
end=`date +%s`
echo ALL running time: $((end-start)) >> bash/log/ABiNENegbatch.txt

start=`date +%s`
python model/train.py --train-data data/mooc/months1/sample2/ratings_train.csv --test-data data/mooc/months1/sample2/ratings_test.csv --alpha 0.01 --beta 0.01 --gamma 0.2 --lam 0.01 --lr 0.01 --ws 5 --ns 4 --walk-length 80 --number-walks 3 --max-iter 200 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNENegbatch.txt
end=`date +%s`
echo ALL running time: $((end-start)) >> bash/log/ABiNENegbatch.txt

start=`date +%s`
python model/train.py --train-data data/mooc/months1/sample3/ratings_train.csv --test-data data/mooc/months1/sample3/ratings_test.csv --alpha 0.01 --beta 0.01 --gamma 0.2 --lam 0.01 --lr 0.01 --ws 5 --ns 4 --walk-length 80 --number-walks 3 --max-iter 200 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNENegbatch.txt
end=`date +%s`
echo ALL running time: $((end-start)) >> bash/log/ABiNENegbatch.txt

start=`date +%s`
python model/train.py --train-data data/mooc/months1/sample4/ratings_train.csv --test-data data/mooc/months1/sample4/ratings_test.csv --alpha 0.01 --beta 0.01 --gamma 0.2 --lam 0.01 --lr 0.01 --ws 5 --ns 4 --walk-length 80 --number-walks 3 --max-iter 200 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNENegbatch.txt
end=`date +%s`
echo ALL running time: $((end-start)) >> bash/log/ABiNENegbatch.txt

start=`date +%s`
python model/train.py --train-data data/mooc/months1/sample5/ratings_train.csv --test-data data/mooc/months1/sample5/ratings_test.csv --alpha 0.01 --beta 0.01 --gamma 0.2 --lam 0.01 --lr 0.01 --ws 5 --ns 4 --walk-length 80 --number-walks 3 --max-iter 200 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNENegbatch.txt
end=`date +%s`
echo ALL running time: $((end-start)) >> bash/log/ABiNENegbatch.txt

start=`date +%s`
python model/train.py --train-data data/mooc/months1/sample6/ratings_train.csv --test-data data/mooc/months1/sample6/ratings_test.csv --alpha 0.01 --beta 0.01 --gamma 0.2 --lam 0.01 --lr 0.01 --ws 5 --ns 4 --walk-length 80 --number-walks 3 --max-iter 200 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNENegbatch.txt
end=`date +%s`
echo ALL running time: $((end-start)) >> bash/log/ABiNENegbatch.txt

start=`date +%s`
python model/train.py --train-data data/mooc/months1/sample7/ratings_train.csv --test-data data/mooc/months1/sample7/ratings_test.csv --alpha 0.01 --beta 0.01 --gamma 0.2 --lam 0.01 --lr 0.01 --ws 5 --ns 4 --walk-length 80 --number-walks 3 --max-iter 200 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNENegbatch.txt
end=`date +%s`
echo ALL running time: $((end-start)) >> bash/log/ABiNENegbatch.txt

start=`date +%s`
python model/train.py --train-data data/mooc/months1/sample8/ratings_train.csv --test-data data/mooc/months1/sample8/ratings_test.csv --alpha 0.01 --beta 0.01 --gamma 0.2 --lam 0.01 --lr 0.01 --ws 5 --ns 4 --walk-length 80 --number-walks 3 --max-iter 200 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNENegbatch.txt
end=`date +%s`
echo ALL running time: $((end-start)) >> bash/log/ABiNENegbatch.txt

start=`date +%s`
python model/train.py --train-data data/mooc/months1/sample9/ratings_train.csv --test-data data/mooc/months1/sample9/ratings_test.csv --alpha 0.01 --beta 0.01 --gamma 0.2 --lam 0.01 --lr 0.01 --ws 5 --ns 4 --walk-length 80 --number-walks 3 --max-iter 200 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNENegbatch.txt
end=`date +%s`
echo ALL running time: $((end-start)) >> bash/log/ABiNENegbatch.txt

start=`date +%s`
python model/train.py --train-data data/mooc/months1/sample10/ratings_train.csv --test-data data/mooc/months1/sample10/ratings_test.csv --alpha 0.01 --beta 0.01 --gamma 0.2 --lam 0.01 --lr 0.01 --ws 5 --ns 4 --walk-length 80 --number-walks 3 --max-iter 200 --d 128 --top-n 10 --ABRW-topk 30 --ABRW-beta 0.2 >> bash/log/ABiNENegbatch.txt
end=`date +%s`
echo ALL running time: $((end-start)) >> bash/log/ABiNENegbatch.txt
