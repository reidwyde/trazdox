#!/bin/bash
rm -rf *~ *# outputData &> /dev/null
make run
terminal="set terminal pdf font \"Times-New-Roman,18\" lw 1"
ext=pdf
font=23
captions=('Saline' 'Doxorubicin' 'Trastuzumab' 'Doxorubicin 24 hrs prior to Trastuzumab' 'Trastuzumab 24 hrs prior to Doxorubicin' 'Trastuzumab + Doxorubicin')
point_type=(7 5 9 11 13 6)
point='ps 0.75 lc 0'
groups=$(ls -1 mean_std_group_*.dat | wc -l)
for g in $(seq 1 ${groups}); do
    let i=g-1
    rm figura.cmd &> /dev/null
    name=solution_group_${g}
    echo ${terminal} | tee -a figura.cmd &> /dev/null
    echo "set output \"${name}.${ext}\"" | tee -a figura.cmd &> /dev/null
    echo "set border linewidth 3" | tee -a figura.cmd &> /dev/null
    echo "set key ins vert reverse top Left left font \",${font}\"" | tee -a figura.cmd &> /dev/null
    echo "set ylabel \"Tumor Volume (mm^3)\" font \",${font}\"" | tee -a figura.cmd &> /dev/null
    echo "set xlabel \"Time (days)\" font \",${font}\"" | tee -a figura.cmd &> /dev/null
    echo "set ytics font \",${font}\"" | tee -a figura.cmd &> /dev/null
    echo "set xtics 12 font \",${font}\"" | tee -a figura.cmd &> /dev/null
#    echo "set title '${captions[${i}]}' offset 0,-2.75 font \",${font}\"" | tee -a figura.cmd &> /dev/null
    echo -n "plot [][0:] " | tee -a figura.cmd &> /dev/null
    echo -n "'mean_std_group_${g}.dat' u 1:2:3 t'' with yerrorbars lw 2 pt ${point_type[${i}]} ${point}," | tee -a figura.cmd &> /dev/null
    echo -n "'mean_std_group_${g}.dat' u 1:2 t'Data' w lp lw 2 pt ${point_type[${i}]} ${point}," | tee -a figura.cmd &> /dev/null
    echo -n "'tumor_evolution_${g}.txt' u 1:2 t'Model' w lp lw 2 pt ${point_type[${i}]} ps 0.75 lc 7" | tee -a figura.cmd &> /dev/null
    gnuplot "figura.cmd"
    rm figura.cmd &> /dev/null
    pdfcrop ${name}.pdf ${name}t.pdf &> /dev/null
    mv ${name}t.pdf ${name}.pdf
done
#cp *.pdf ../New_Models/figures/
#cd ../New_Models/
#    ./lista.sh
