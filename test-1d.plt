set term png size 640,480
set output "test-1d.png"

plot "test.dat" using 1:2 w lines

set output
