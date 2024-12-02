#!/bin/bash

. ./path.sh || exit 1;

if [ $# != 1 ]; then
  echo "Usage: $0 <audio-path>"
  echo " $0 /home/data/aishell4"
  exit 1;
fi

aishell4_source_dir=$1
train_dir=data/local/aishell4_train
test_dir=data/local/aishell4_test

mkdir -p $train_dir
mkdir -p $test_dir

# data directory check
if [ ! -d $aishell_audio_dir ] || [ ! -f $aishell_text ]; then
  echo "Error: $0 requires two directory arguments"
  exit 1;
fi

# example data:
# ▶ soxi /jfs-hdfs/user/Archive/OpenSourceDataset/ASR/aishell4/test/wav/L_R003S01C02.flac
#    Input File     : '/jfs-hdfs/user/Archive/OpenSourceDataset/ASR/aishell4/test/wav/L_R003S01C02.flac'
#    Channels       : 8
#    Sample Rate    : 16000
#    Precision      : 16-bit
#    Duration       : 00:39:22.95 = 37807120 samples ~ 177221 CDDA sectors
#    File Size      : 286M
#    Bit Rate       : 968k
#    Sample Encoding: 16-bit FLAC
#    Comment        : 'audio_encoder=Lavf57.76.100'
for room_name in "train_L" "train_M" "train_S" "test"; do
  if [ -f ${aishell4_source_dir}/$room_name/wav_list.txt ];then
    rm  ${aishell4_source_dir}/$room_name/wav_list.txt
  fi
  FILES="${aishell4_source_dir}/$room_name/wav/*.flac"
  for f in $FILES; do
    if [ ! -f $f.1channel.wav ];then
      sox $f -c 1 -r 16000 -b 16 $f.1channel.wav
      echo "convert $f to $f.1channel.wav"
    fi
    echo "$f.1channel.wav" >> ${aishell4_source_dir}/$room_name/wav_list.txt
  done
  if [ -f ${aishell4_source_dir}/$room_name/TextGrid_list.txt ];then
    rm ${aishell4_source_dir}/$room_name/TextGrid_list.txt
  fi
  FILES="${aishell4_source_dir}/$room_name/TextGrid/*.TextGrid"
  for f in $FILES; do
    echo "$f" >> ${aishell4_source_dir}/$room_name/TextGrid_list.txt
  done
done

rm -rf ${aishell4_source_dir}/full_train
mkdir -p ${aishell4_source_dir}/full_train
for r in train_L train_M train_S ; do
  cat ${aishell4_source_dir}/$r/TextGrid_list.txt >> ${aishell4_source_dir}/full_train/textgrid.flist
  cat ${aishell4_source_dir}/$r/wav_list.txt >> ${aishell4_source_dir}/full_train/wav.flist
done

# process train set
cp ${aishell4_source_dir}/full_train/wav.flist $train_dir
cp ${aishell4_source_dir}/full_train/textgrid.flist $train_dir
sed -e 's/\.flac\.1channel\.wav//' $train_dir/wav.flist | awk -F '/' '{print $NF}' > $train_dir/utt.list
paste -d' ' $train_dir/utt.list $train_dir/wav.flist | sort -u > $train_dir/wav.scp
python local/aishell4_process_textgrid.py --path $train_dir
cat $train_dir/text_all | local/text_normalize.pl | local/text_format.pl | sort -u > $train_dir/text
local/filter_scp.pl -f 1 $train_dir/text $train_dir/utt2spk_all | sort -u > $train_dir/utt2spk
local/utt2spk_to_spk2utt.pl $train_dir/utt2spk > $train_dir/spk2utt
local/filter_scp.pl -f 1 $train_dir/text $train_dir/segments_all | sort -u > $train_dir/segments

# process test set
cp ${aishell4_source_dir}/test/wav_list.txt $test_dir/wav.flist
cp ${aishell4_source_dir}/test/TextGrid_list.txt $test_dir/textgrid.flist
sed -e 's/\.flac\.1channel\.wav//' $test_dir/wav.flist | awk -F '/' '{print $NF}' > $test_dir/utt.list
paste -d' ' $test_dir/utt.list $test_dir/wav.flist |sort -u > $test_dir/wav.scp
python local/aishell4_process_textgrid.py --path $test_dir
cat $test_dir/text_all | local/text_normalize.pl | local/text_format.pl | sort -u > $test_dir/text
local/filter_scp.pl -f 1 $test_dir/text $test_dir/utt2spk_all | sort -u > $test_dir/utt2spk
local/utt2spk_to_spk2utt.pl $test_dir/utt2spk > $test_dir/spk2utt
local/filter_scp.pl -f 1 $test_dir/text $test_dir/segments_all | sort -u > $test_dir/segments

local/copy_data_dir.sh --utt-prefix Aishell4- --spk-prefix Aishell4- \
  $train_dir data/aishell4_train
local/copy_data_dir.sh --utt-prefix Aishell4- --spk-prefix Aishell4- \
  $test_dir data/aishell4_test

echo "$0: AISHELL4 data preparation succeeded"
exit 0;
