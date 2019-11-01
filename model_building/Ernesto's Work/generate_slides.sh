#!/bin/bash
clear
file="slides"
pdflatex ${file}.tex
pdflatex ${file}.tex
rm *~ *.blg *.bbl *.toc *.snm *.out *.nav *.aux *.log &> /dev/null
( evince ${file}.pdf ) &
