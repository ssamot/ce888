#!/bin/bash

echo "Building index html"
#pandoc -s -S --toc --template=custom.html -c "./labs.css" index.md -o index.html
cat index.md suggestions.md > tmp.md
pandoc tmp.md -o index.html --template template.html --toc --toc-depth 2

#echo "Building html for suggestions"
#pandoc suggestions.md --template template.html  -o ./labs/suggestions/suggestions.html --variable suggestions=True --variable topdir=../..


#for lec_name in "01-Introduction" "02-Stats" "03-Modelling" "04-Bandits" "05-Recommender" "06-Exploration" "07-Neural" "08-Text"
 for lec_name in causal
 	do
 	echo Building slides for lecture $lec_name
 	pandoc  --slide-level 2 --template=custom.beamer -V theme=bjeldbak -V linkcolor=blue --toc -t beamer $lec_name.md -o ./slides/$lec_name-slides.pdf
 	pdfnup ./slides/$lec_name-slides.pdf -q --nup 2x2 --noautoscale false --delta "0.2cm 0.3cm" --frame true --scale 0.95 -o ./slides/$lec_name-handouts.pdf
	done


#for lab_name in "1" "2" "3" "4" "5" "6" "7" "8"
# for lab_name in "8"
# do
# 	echo "Building html for lab "$lab_name
# 	pandoc -s -S --toc --template=custom.html -c "../../labs.css" 0$lab_name-labs.md -o ./labs/lab$lab_name/0$lab_name-labs.html
# done

rm tmp.md