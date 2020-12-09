#! /bin/bash
/opt/amazon/openmpi/bin/mpirun -np 2 -H \
    ip-172-31-88-106:1,ip-172-31-76-23:1 \
    -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH \
    -x HOROVOD_LOG_LEVEL=ERROR \
    -tag-output \
    -mca pml ob1 \
    -mca btl ^openib \
    -mca btl_tcp_if_exclude lo,docker0 \
    /home/ubuntu/anaconda3/envs/horovod_dev/bin/python /home/ubuntu/horovod-examples/pytorch/basic_tests.py 