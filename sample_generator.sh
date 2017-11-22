

export ck_dir='/home/kuznetsov/tmp/thread31/'
# export index_list="${ck_dir}/ckpt_ids.txt"
export ckpt=243000
export iteration_number=5
export dir="${ck_dir}"
export res_dir="${dir}/results/"
mkdir ${res_dir}
for i in {0..49}
do
echo "Start $i iteration"
# python -m nmt.nmt --out_dir="${dir}/model" --ckpt="${dir}model/translate.ckpt-${line}" --inference_input_file="${dir}/data/d_train.cor" --inference_output_file="${res_dir}d_train-${line}.nmt"
python -m nmt.nmt --out_dir="${dir}/model" --ckpt="${dir}model/translate.ckpt-${ckpt}" --inference_input_file="${dir}/data/d_train.cor" --inference_output_file="${res_dir}d_train-${ckpt}-${i}.nmt"
done
# done < $index_list
