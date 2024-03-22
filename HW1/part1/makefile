PY = python3

test:
	$(PY) eval.py

main:
	# $(PY) main.py --threshold=3.0 --image_path="./testdata/1.png"
	$(PY) main.py --threshold=1.0 --image_path="./testdata/2.png"
	$(PY) main.py --threshold=2.0 --image_path="./testdata/2.png"
	$(PY) main.py --threshold=3.0 --image_path="./testdata/2.png"

clean:
	rm -f *.png
	rm -rf __pycache__