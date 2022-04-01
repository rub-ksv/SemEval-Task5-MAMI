#!/bin/bash -u


datasetdir=./MAMI	# The data source dir, you should download dataset and save it under this dir (e.g. MAMI/training and MAMI/test)
datadir=./Dataset	# The dir where saves the processed data and features
modeldir=./model	# The dir saves the trained models	
results=./results	# The dir saves the results, which is predicted by the trained model. The evaluation results are also saved here
ifgpu=false		# ifgpu is true, then using GPU for training
ifdebug=false		# ifdebug is true, when the test set haven't released. Split a part of training set as the test set
num=700			# ifdebug is true, split $num samples from training set as the test set
best_model=acc		# we can selected choose best accuracy model or best loss model		
adjdir=./		# the dir save GCAN adjacency matrix
stage=0			# which running process should be started
stop_stage=0		# stop at which process

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
	### split MEMEs	
	for dset in training test; do
	     python3 local/orgimagesplit.py $datasetdir $dset || exit 1;
	done

	### using image caption to extract the image-text description (https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
	for dset in training test; do ## if the test set relased, here can add for dset in training test; do
	    savedir=$datadir/imagecap
	    capdir=local/imagecap
	    python3 local/imagecap/caption_multisplit.py $datasetdir $dset $savedir $capdir $ifgpu || exit 1;
	done
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

	### train text model
	tasktype=classification
	featype=bert
	iffinetune=true

	stream=ocr
	for i in 0 1 2 3 4 5 6 7 8 9; do # start from $i folder in 10 folder cross-validation.
	python3 ./local/train.py $datadir $tasktype $featype $modeldir $results $stream $best_model $ifgpu $iffinetune $i $adjdir || exit 1;
	done

	### train image model
	stream=image
	i=0 # start from $i folder in 10 folder cross-validation	
	python3 ./local/train.py $datadir $tasktype $featype $modeldir $results $stream $best_model $ifgpu $iffinetune $i $adjdir || exit 1;

	
	stream=BERTC_bi ## BERTC-VIT model 
	for i in 0 1 2 3 4 5 6 7 8 9; do
	python3 ./local/train.py $datadir $tasktype $featype $modeldir $results $stream $best_model $ifgpu $iffinetune $i $adjdir || exit 1;
	done

	stream=GCAN_bi ## BERTC-VIT model 
	for i in 0 1 2 3 4 5 6 7 8 9; do
	python3 ./local/train.py $datadir $tasktype $featype $modeldir $results $stream $best_model $ifgpu $iffinetune $i $adjdir || exit 1;
	done

	stream=bi_BERTC_GCAN ## BERTC-GCAN model 
	for i in 0 1 2 3 4 5 6 7 8 9; do
	python3 ./local/train.py $datadir $tasktype $featype $modeldir $results $stream $best_model $ifgpu $iffinetune $i $adjdir || exit 1;
	done

	stream=bi ## BERTC-GCAN-Vit model 
	for i in 0 1 2 3 4 5 6 7 8 9; do
	python3 ./local/train.py $datadir $tasktype $featype $modeldir $results $stream $best_model $ifgpu $iffinetune $i $adjdir || exit 1;
	done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
	#ensemble
	ls -R ./model/classification/ > ./classification.txt
	python3 ./local/ensemble.py $results $datasetdir/test_labels.txt ./classification.txt #$datasetdir/test_labels.txt: the grund turth of test set


fi
