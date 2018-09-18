#!/bin/sh

ncall=1000
itmx=10

if [ $# -gt 0 ];then
    ncall=$1
else
    echo $0 ncall itmx nblk
    exit
fi
if [ $# -gt 1 ];then
    itmx=$2
fi
if [ $# -gt 2 ];then
    nblk=$3
fi

echo run: gVegasDouble ${ncall} %{itmx} ${nblk}

gVegasDouble -n=${ncall} -i=${itmx} -b=${nblk}

