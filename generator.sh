

export ck_dir='/home/kuznetsov/tmp/genthread/'
export index_list="${ck_dir}/ckpt_ids.txt"
export dir="${ck_dir}"
export res_dir="${dir}/results/"
mkdir ${res_dir}
while read line;
do
echo $line
python -m nmt.nmt --out_dir="${dir}/model" --ckpt="${dir}model/translate.ckpt-${line}" --inference_input_file="${dir}/data/d_train.cor" --inference_output_file="${res_dir}d_train-${line}.nmt"
done < $index_list
