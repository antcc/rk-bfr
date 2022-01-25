$pdflatex = 'pdflatex -shell-escape -file-line-error -interaction=nonstopmode -quiet -synctex=1 %O %S';
$pdf_previewer = 'sioyek';

#automatically call lualatex/pdflatex (instead of latex)
$pdf_mode = 1;

# enable deletion of *.bbl when calling "latexmk -c"
$bibtex_use = 2;

#remove more files than in the default configuration
@generated_exts = qw(acn acr alg aux code ist fls glg glo gz gls glsdefs idx ind lof lot out thm toc tpt wrt run.xml nav snm synctex  synctex.gz mw bcf synctex(busy));
$clean_ext .= ' %R.ist %R.xdy';
