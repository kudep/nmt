

export ck_dir='/home/kuznetsov/tmp/thread31/'
export ckpt=243000
export dir="${ck_dir}"
export res_dir="${dir}/results/"
mkdir ${res_dir}
python -m nmt.nmt --out_dir="${dir}/model" --ckpt="${dir}model/translate.ckpt-${ckpt}" --inference_input_file="${dir}/data/train.cor" --inference_output_file="${res_dir}train-${ckpt}.nmt"
