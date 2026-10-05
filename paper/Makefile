# Build the paper: `make` -> main.pdf ; `make arxiv` -> arxiv.tar.gz ready for upload
main.pdf: main.tex references.bib
	latexmk -pdf -interaction=nonstopmode main.tex

arxiv: main.pdf
	tar czf arxiv.tar.gz main.tex main.bbl references.bib

clean:
	latexmk -C
	rm -f arxiv.tar.gz

.PHONY: arxiv clean
