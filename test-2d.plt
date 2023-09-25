set term png size 640,480
set output "test-2d.png"

set autoscale xfix
set autoscale yfix
set autoscale cbfix

plot "test2d.dat" matrix nonuniform w image notitle

set output
