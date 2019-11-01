#!/bin/bash
rm -rf *~ *# outputData error.txt error_g*.txt script.py &> /dev/null
make run
terminal="set terminal pdf font \"Times-New-Roman,18\" lw 1"
ext=pdf
font=23
captions=('Saline' 'Doxorubicin' 'Trastuzumab' 'Doxorubicin 24 hrs prior to Trastuzumab' 'Trastuzumab 24 hrs prior to Doxorubicin' 'Trastuzumab + Doxorubicin')
point_type=(7 5 9 11 13 6)
point='ps 0.75 lc 0'
groups=$(ls -1 mean_std_group_*.dat | wc -l)
touch error.txt
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
    echo -n "plot [0:][0:] " | tee -a figura.cmd &> /dev/null
    echo -n "'mean_std_group_${g}.dat' u 1:2:3 t'' with yerrorbars lw 2 pt ${point_type[${i}]} ${point}," | tee -a figura.cmd &> /dev/null
    echo -n "'mean_std_group_${g}.dat' u 1:2 t'Data' w lp lw 2 pt ${point_type[${i}]} ${point}," | tee -a figura.cmd &> /dev/null
    echo -n "'tumor_evolution_${g}.txt' u 1:2 t'Model' w lp lw 2 pt ${point_type[${i}]} ps 0.75 lc 7" | tee -a figura.cmd &> /dev/null
    gnuplot "figura.cmd"
    rm figura.cmd tmp.txt tmp2.txt &> /dev/null
    pdfcrop ${name}.pdf ${name}t.pdf &> /dev/null
    mv ${name}t.pdf ${name}.pdf
    paste tumor_evolution_${g}.txt mean_std_group_${g}.dat > error_g${g}.txt
    more error_g${g}.txt | awk 'function abs(v) {return v < 0 ? -v : v}{printf "%e\n",100.0*abs($2-$4)/$4}' > error_g${g}p.txt
    sed -n -e "1,19p" error_g${g}p.txt > tmp.txt
    mv tmp.txt error_g${g}p.txt
    paste -d" " error.txt error_g${g}p.txt > tmp.txt
    mv tmp.txt error.txt
done
more error.txt | sed 's/ *$//' > tmp.txt
mv tmp.txt error.txt
echo "import sys" | tee -a script.py &> /dev/null
echo "import pandas as pd" | tee -a script.py &> /dev/null
echo "data = pd.read_csv('error.txt',header=None,delimiter=' ')" | tee -a script.py &> /dev/null
echo "orig_stdout = sys.stdout" | tee -a script.py &> /dev/null
echo "f = open('error_m_sd.txt', 'w')" | tee -a script.py &> /dev/null
echo "sys.stdout = f" | tee -a script.py &> /dev/null
echo "for column in data:" | tee -a script.py &> /dev/null
echo "    print data[column].mean(),data[column].std()" | tee -a script.py &> /dev/null
echo "sys.stdout = orig_stdout" | tee -a script.py &> /dev/null
echo "f.close()" | tee -a script.py &> /dev/null
pvpython script.py
sed -n -e "2,7p" error_m_sd.txt > tmp.txt
#tac error_m_sd.txt > tmp.txt
mv tmp.txt error_m_sd.txt
NL=$(more error_m_sd.txt | wc -l)
echo "Lines = ${NL}"
echo "\begin{figure}[!htbp]"
echo "\centering"
for l in $(seq 1 ${NL}); do
    mean=$(sed -n -e "${l}p" error_m_sd.txt  | awk '{printf "%.2f",$1-0}')
    stdv=$(sed -n -e "${l}p" error_m_sd.txt  | awk '{printf "%.2f",$2-0}')
    echo "\begin{subfigure}{0.32\textwidth}"
    echo "\centering"
    echo "\includegraphics[width=\textwidth]{exp/solution_group_${l}}\caption{\$${mean}\pm ${stdv}\%\$}"
    echo "\end{subfigure}"
done
echo "\end{figure}"
#cp *.pdf ../New_Models/figures/exp/
#cd ../New_Models/
#./lista.sh
