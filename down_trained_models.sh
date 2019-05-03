DIR_NAME="/tmp/"

cd $DIR_NAME

# svm
MODEL_FNAME="svm_grasp_baseline_iter1000.pkl"

if [ ! -f $DIR_NAME/$MODEL_FNAME ]; then
    wget http://people.csail.mit.edu/cchoi/dataset/grasping/$MODEL_FNAME
else
    echo "$MODEL_FNAME exists. Skip downloading."
fi

# 3dcnn
MODEL_FNAME="grasping_epoch150_train_noside_cnn3d.h5"

if [ ! -f $DIR_NAME/$MODEL_FNAME ]; then
    wget http://people.csail.mit.edu/cchoi/dataset/grasping/$MODEL_FNAME
else
    echo "$MODEL_FNAME exists. Skip downloading."
fi

MODEL_FNAME="grasping_epoch150_train_cnn3d_side.h5"

if [ ! -f $DIR_NAME/$MODEL_FNAME ]; then
    wget http://people.csail.mit.edu/cchoi/dataset/grasping/$MODEL_FNAME
else
    echo "$MODEL_FNAME exists. Skip downloading."
fi

MODEL_FNAME="grasping_epoch1000_train_fcn.h5"

if [ ! -f $DIR_NAME/$MODEL_FNAME ]; then
    wget http://people.csail.mit.edu/cchoi/dataset/grasping/$MODEL_FNAME
else
    echo "$MODEL_FNAME exists. Skip downloading."
fi

cd -