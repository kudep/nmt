

export ck_dir='/home/kuznetsov/tmp/thread31/'
# export index_list="${ck_dir}/ckpt_ids.txt"
export ckpt=457000
export dir="${ck_dir}"
export res_dir="${dir}/test/"
for i in {0..4}
do
echo "Start $i iteration"
# python -m nmt.nmt --out_dir="${dir}/model" --ckpt="${dir}model/translate.ckpt-${line}" --inference_input_file="${dir}/data/d_train.cor" --inference_output_file="${res_dir}d_train-${line}.nmt"
python -m nmt.nmt --out_dir="${dir}/model" --ckpt="${dir}model/translate.ckpt-${ckpt}" --inference_input_file="${dir}/data/d_train_tail.cor" --inference_output_file="${res_dir}d_train-${ckpt}-${i}.nmt"
done
# done < $index_list
