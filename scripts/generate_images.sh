# $1 is the tristan output directory
# $2 is the path to the deposit_particles binary

echo $1 $2

n=$(ls -l $1 |rg prtl |sort |tail -n1 |sed 's/prtl.tot.//g')
ppcs=(1600 32 16 4)
for ppc in "${ppcs[@]}"; do
  for i in $(seq 1 $n); do
    thin=$((1600 / ppc))
    $2 --field-file $1/flds.tot.$i --particle-file $1/prtl.tot.$i --param-file $1/param.$1 --thin $thin --output-file out-$ppc.$i
    echo done $i of $n for ppc=$ppc
  done
done
