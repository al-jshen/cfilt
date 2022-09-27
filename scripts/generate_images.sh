#!/bin/bash
# $1 is the tristan output directory
# $2 is the path to the deposit_particles binary

n=$(ls $1 |rg prtl |sort |tail -n1 |sed 's/prtl.tot.//g')
ppcs=(1600 320 160 80 32 16 4 1)
for ppc in "${ppcs[@]}"; do
  for i in $(seq -w 1 $n); do
    thin=$((1600 / ppc))
    fname=out-$ppc.$i
    if [ -f "$fname" ]; then
	    echo skipping $i of $n for ppc=$ppc
    else
	    $2 --field-file $1/flds.tot.$i --particle-file $1/prtl.tot.$i --param-file $1/param.$i --thin $thin --output-file $fname &
      echo done $i of $n for ppc=$ppc
    fi
  done
done
