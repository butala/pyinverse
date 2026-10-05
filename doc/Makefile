# Build and check the 3-D FBP write-up.
#
#   make pdf       typeset the PDF from the org file
#   make check     build, then verify the LaTeX log and the key content
#   make md        export a GitHub-renderable markdown copy
#   make date      stamp #+DATE: (done automatically by `pdf`)
#   make clean     drop LaTeX by-products and the ltximg/ preview cache
#   make view      open the write-up in Emacs (non-batch)
#
# doc/fbp3-filter-derivation.org is the source of truth.  There is no markdown
# original -- it was retired on 2026-09-18 -- so edit the .org directly and
# rebuild.  `make md` is the *reverse* route: it turns the .org into a
# GitHub-renderable .md for reading in the repository browser; edit the .org,
# never the .md.  The two org parsing traps that break a LaTeX export are
# documented at the top of the .org file itself:
#
#   * a continuation line starting with "+" or "-" is read as a list item, so
#     display equations must stay on one line;
#   * a "|" inside a table cell is a column separator (use \lvert f_u\rvert).
#
# #+DATE: is generated (it used to be manual and went stale): `make date`
# stamps it with the date of the last commit that touched the .org, or with
# today when the file has uncommitted edits.  `pdf` runs it, so the date on the
# title page tracks the content.
#
# The export deliberately goes through org-latex-export-to-latex and then runs
# pdflatex here, rather than using org-latex-export-to-pdf: the latter compiles
# in a scratch directory, so no .log is left beside the source for `check` to
# inspect, and a single extra pdflatex pass on a source with no .toc yields a
# PDF whose table of contents is empty.  Three passes: the first writes the
# .toc, the rest settle the page numbers.  Org's own "PDF file produced with
# errors" message refers to its first of several passes and is not reliable,
# so `check` reads the .log instead.

DOC := fbp3-filter-derivation

.PHONY: pdf md date view check clean

date:
	@d=$$(git log -1 --format=%cs -- $(DOC).org 2>/dev/null); \
	 if ! git diff --quiet -- $(DOC).org 2>/dev/null; then d=$$(date +%F); fi; \
	 if [ -z "$$d" ]; then d=$$(date +%F); fi; \
	 awk -v d="$$d" 'index($$0, "#+DATE:") == 1 { print "#+DATE: " d; next } { print }' \
	     $(DOC).org > $(DOC).org.tmp && mv $(DOC).org.tmp $(DOC).org; \
	 echo "#+DATE: $$d"

pdf: date $(DOC).org
	rm -f $(DOC).tex
	emacs --batch -l ox-latex $(DOC).org -f org-latex-export-to-latex >/dev/null 2>&1 || true
	@test -f $(DOC).tex || { echo "FAIL: org export produced no .tex (see: emacs --batch -l ox-latex $(DOC).org -f org-latex-export-to-latex)"; exit 1; }
	pdflatex -interaction=nonstopmode $(DOC).tex >/dev/null 2>&1 || true
	pdflatex -interaction=nonstopmode $(DOC).tex >/dev/null 2>&1 || true
	pdflatex -interaction=nonstopmode $(DOC).tex >/dev/null 2>&1 || true

md: date $(DOC).org
	rm -f $(DOC).md
	emacs --batch -l ox-md $(DOC).org -f org-md-export-to-markdown >/dev/null 2>&1 || true
	@test -f $(DOC).md || { echo "FAIL: org -> markdown produced no .md (see: emacs --batch -l ox-md $(DOC).org -f org-md-export-to-markdown)"; exit 1; }
	@echo "OK: $(DOC).md written"

view: $(DOC).org
	emacs $(DOC).org &

check: pdf md
	@test -f $(DOC).log || { echo "FAIL: no LaTeX log -- the export did not run"; exit 1; }
	@! grep -q '^!' $(DOC).log || { echo "FAIL: LaTeX errors in the export:"; \
	   grep -A4 '^!' $(DOC).log | head -20; exit 1; }
	@! grep -q 'undefined' $(DOC).log || { echo "FAIL: undefined references"; \
	   grep -i 'undefined' $(DOC).log | head; exit 1; }
	pdftotext $(DOC).pdf /tmp/$(DOC).txt
	@! grep -q 'backslash' /tmp/$(DOC).txt || \
	   { echo "FAIL: math was exported as escaped plain text"; exit 1; }
	@for s in Contents Jacobian "missing cone" "slice decoupling" 0.39129 3.8 Grangeat \
	          "sec" "0.816" "cone fraction" "tilted" "plateaus"; do \
	   grep -q "$$s" /tmp/$(DOC).txt || { echo "MISSING from the PDF: $$s"; exit 1; }; \
	done
	@for s in Jacobian "missing cone" Grangeat "cone fraction" "plateaus"; do \
	   grep -q "$$s" $(DOC).md || { echo "MISSING from the markdown: $$s"; exit 1; }; \
	done
	@grep -q '^#+DATE: [0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}$$' $(DOC).org || \
	   { echo "FAIL: #+DATE: is not an ISO date"; exit 1; }
	@echo "OK: org -> PDF builds cleanly and carries the expected content"
	@echo "OK: org -> markdown carries the expected content"

clean:
	rm -f $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc $(DOC).tex
	rm -rf ltximg
